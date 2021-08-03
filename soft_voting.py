import uuid
import argparse
import pandas as pd
import os
import numpy as np

import json
import datetime

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dump_json(fn, json_obj):
    # Write json_obj to a file named fn

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__

    with open(fn, 'w') as fp:
        json.dump(json_obj, fp, indent=4, cls=NumpyEncoder)


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-s', '--run-settings', default='default_settings.json')
    args = parser.parse_args()

    def combine_and_write(probas_fns, out_dir, id_col, target_col, classes, make_preds=True):

        runname = str(uuid.uuid4())

        combined = {}
        combined['runname'] = runname
        combined['startime'] = str(datetime.datetime.now())
        combined['probas_fns'] = probas_fns
        combined['out_dir'] = out_dir

        print("Combining files: {}".format(probas_fns))

        # Read one of the files to get the IDs
        combined_df = pd.read_csv(probas_fns[0], usecols=[id_col])

        probas = list()
        for probas_fn in probas_fns:
            _df = pd.read_csv(probas_fn)
            probas.append(_df.drop(id_col, axis=1))

        mean_df = pd.DataFrame(np.mean(probas, axis=0), columns=classes)

        combined_df = pd.concat([combined_df, mean_df], axis=1)


        if make_preds:
            combined_df[target_col] = combined_df[classes].idxmax(axis=1)
            print(combined_df[target_col].value_counts())

        else:
            # leave probas
            combined_df[target_col] = combined_df[classes[-1]]

        print(combined_df.head())

        out_fn = os.path.join(out_dir, 'combined-{}-preds.csv'.format(runname))
        combined['out_fn'] = out_fn

        combined_df[[id_col, target_col]].to_csv(out_fn, index=False)
        print("Wrote probas file {}".format(out_fn))

        json_fn = os.path.join(out_dir, "combined-{}.json".format(runname))
        dump_json(json_fn, combined)

        return out_fn, runname
    
    do_pump = True
    do_vaccine = False
    do_earthquake = False

    ##########################
    # H1N1
    ##########################

    if do_vaccine:
        # One way is to just get the top N results from the results file
        num_top = 9
        results = pd.read_csv("h1n1/tune_results.csv").head(num_top)
        probas_fns = list(results['probas_fn'])

        # Another way is to manually list them
        probas_fns = [
            'h1n1/out/82695d7e-1316-4705-babb-7d3d61f6db19-000-probas.csv',
            #'h1n1/out/458d5f7e-fcda-4e00-9f49-bde39cf70726-000-probas.csv',
            #'h1n1/out/f27193d9-af1a-4daa-a225-e4ce133c314b-000-probas.csv',
        ]

        h1n1_out_fn, h1n1_runname = combine_and_write(
            probas_fns,
            "h1n1/out",
            "respondent_id",
            "h1n1_vaccine",
            classes= ["0", "1"],
            make_preds = False
        )

    ##########################
    # Seasonal
    ##########################

        # One way is to just get the top N results from the results file
        num_top = 9 
        results = pd.read_csv("seasonal/tune_results.csv").head(num_top)
        probas_fns = list(results['probas_fn'])

        # Another way is to manually list them
        probas_fns = [
            'seasonal/out/388ef4f3-cdb8-4729-b0f9-7d93183d12fa-000-probas.csv'
        ]

        seasonal_out_fn, seasonal_runname = combine_and_write(
            probas_fns,
            "seasonal/out",
            "respondent_id",
            "seasonal_vaccine",
            classes= ["0", "1"],
            make_preds = False
        )

    #################################
    # Now combine H1N1 and Seasonal
    #################################


        h1n1_df = pd.read_csv(h1n1_out_fn)
        seasonal_df = pd.read_csv(seasonal_out_fn)

        assert(h1n1_df['respondent_id'].equals(seasonal_df['respondent_id']))

        vaccine_df = pd.DataFrame(
            {'respondent_id': h1n1_df['respondent_id'],
             'h1n1_vaccine': h1n1_df['h1n1_vaccine'],
             'seasonal_vaccine': seasonal_df['seasonal_vaccine']
            })

        print(vaccine_df.head())

        new_fn = "{}_{}_vaccine-combined-preds.csv".format(h1n1_runname, seasonal_runname)
        print("Writing vaccine combined filename: {}".format(new_fn))
        vaccine_df.to_csv("vaccine/out/{}".format(new_fn), index=False)



    ##########################
    # Pump
    ##########################

    if do_pump:

        # One way is to just get the top N results from the results file
        num_top = 5
        results = pd.read_csv("pump/tune_results.csv").head(num_top)
        probas_fns = list(results['probas_fn'])

        # Another way is to manually list them
        probas_fns.extend([
            "pump/out.bak3/d51fe137-1c5a-4447-b9c7-6f44faf4bbcd-9862b73d-21a3-4087-ba1a-66dd621caaaa-probas.csv",
            "pump/out.bak3/a9075359-096e-4372-bd86-1dec64528603-d4039d45-2296-4548-9071-d555dc163da1-probas.csv",
            "pump/out.bak3/ae3d35d3-ff44-4896-ba2c-aaea0e44eb92-02c6ffd1-31ae-49ca-826e-cd8e7579f776-probas.csv",
            "pump/out.bak3/5aa521f4-7021-4b92-b110-5683175b07c9-c0f1d1e8-8a31-49d3-89fd-fb8d1afa5397-probas.csv",
            "pump/out.bak3/91e77532-9c3f-4213-8a2f-02a46ea4ab5d-17fda2ae-b64f-44a4-a629-f9a96239a022-probas.csv",
            "pump/out.bak3/e54de502-7b85-47e4-9a24-29483bf1f680-e99a6d5c-99d8-4e39-a070-a3ae8acc212a-probas.csv",
            "pump/out.bak3/f70e798b-8b59-4894-b77a-3978825cfdd3-76a9bee4-5a2f-4bab-b6a9-5b29d34785a0-probas.csv",
            "pump/out.bak3/9a6b9462-4bea-4b91-bf73-602813d20d40-e7c0e032-9b6a-451e-b9ef-771db079956c-probas.csv",
            "pump/out.bak3/7d3b4257-d1f6-436b-afcd-d6729dacbcf9-23e6f5e1-5ff4-4b35-9107-dfe5341042b0-probas.csv",
            "pump/out.bak3/99614452-ff39-4c80-9747-80944689b559-e07e2046-e4ce-4fa9-aaca-8bbda541b8bc-probas.csv",
            "pump/out.bak3/99614452-ff39-4c80-9747-80944689b559-d2b16e95-5ac7-4368-9217-7789d9979f94-probas.csv",
            "pump/out.bak3/27373ef9-2fca-4554-9896-4a57a8b3215a-283dfb3b-0a6b-4f5d-ad09-a77253dab67e-probas.csv",
            "pump/out.bak3/7f2f7495-8d27-4cea-bcf9-383488da2254-7a720304-b6ea-471b-a5d3-260fcfea45b8-probas.csv",
            "pump/out.bak3/2e6e2fed-1cc8-4f59-ad62-5bd2ffe57008-0e5727fe-32a0-4a3d-9ff5-bfa4baee74d2-probas.csv",
            "pump/out.bak3/ae6a5960-3ab0-4241-8f88-ed0c3b06f6b6-245c9e86-3361-492f-b957-4dd72083f913-probas.csv",
            "pump/out.bak3/3acad507-8ee2-4491-b620-d23f1a2f6063-fd4d1a26-e762-491a-9da9-222df8a1a263-probas.csv",
            "pump/out.bak3/b4b0c534-e5fd-4bdd-b918-ac730de6a850-5359c734-bd0c-4b5a-bbf5-9e40a6fb9400-probas.csv",
            "pump/out.bak3/dac03a6f-042d-48b8-9720-156fd8a880ca-6b606f96-a5b5-4a4a-b0fe-3f5d7c7bbf4f-probas.csv",
            "pump/out.bak3/c06321ea-7ecb-4626-b061-cb52567a6c63-4218c7de-8340-4a42-ad13-50e06036e99d-probas.csv",
            "pump/out.bak3/a515ec3e-1f24-4c9b-b04d-8007c644da5a-0e537295-df4a-434a-8758-1290fb06ec5f-probas.csv",
            #"pump/out/671be414-4717-4306-8052-e665691e053d-a2738273-7bd0-4c24-9aa7-1659319c0ea0-probas.csv",
            #"pump/out/91e77532-9c3f-4213-8a2f-02a46ea4ab5d-364c3cb8-c346-49ca-910e-d20698ceba54-probas.csv",
            #"pump/out/cc7244a5-9d2d-47c7-b43b-2915ff6d4c2b-a1872063-0fa6-412e-9185-bbc0a017d9d7-probas.csv",
            #"pump/out/8cf97119-ddf9-4ea7-8efb-9d19643832c9-ce71b839-4efe-4970-ae2b-c68f3ef84049-probas.csv",
            #"pump/out/9143bb59-0b2c-40ab-9368-4e93b4de29ee-be393556-8610-40d8-96d6-bbe530a8bcb0-probas.csv",
            #"pump/out/d7ce5dfe-8763-476a-aae3-37f73bd7afd5-3fd22796-c5f2-4c33-a901-cfce5f424082-probas.csv",
            #"pump/out/985d1069-2741-415c-9f01-ee9330fe4bf2-36cb0f9a-10bb-4084-8ec5-e152d785eab3-probas.csv",
            #"pump/out/5b550076-1233-446b-89b3-fcef04f0f7da-0dbf0089-84ac-4766-a2db-c134a71fdc5b-probas.csv",
            #"pump/out/1885be1e-c97f-461e-9987-37a6b606999b-164243e4-e0c2-4e23-97df-de16337202d5-probas.csv",
            #"pump/out/f0b9059e-d307-41b4-aa94-4639524d8d4e-8ce5357a-ebae-4179-acab-30f0a08d3cc9-probas.csv"

        )

        _, _ =  combine_and_write(
            probas_fns,
            "pump/out",
            "id",
            "status_group",
            classes= ['functional', 'functional needs repair', 'non functional'],
            make_preds = True
        )

    ##########################
    # Earthquake
    ##########################

    if do_earthquake:

        # One way is to just get the top N results from the results file
        num_top = 9
        results = pd.read_csv("earthquake/tune_results.csv").head(num_top)
        probas_fns = list(results['probas_fn'])

        # Another way is to manually list them
        #probas_fns = ['earthquake/out/5db861e6-c1ce-40e3-91c9-f47c35c65354-411f5491-c145-4f4c-802f-9cc7d61ba51e-probas.csv',
                     #'earthquake/out/d0b7a1ff-da01-41c6-ba80-51a01a407b63-dcc9ccbc-1449-403e-b681-c3320fbc636c-probas.csv',
                     #'earthquake/out/a635d3c9-a808-4cd0-8f92-e8b6be858a96-9d83a7d7-a748-49a0-ace0-4149c54a3ffc-probas.csv',
                     #'earthquake/out/c5bc08c3-0ec8-4d82-8dfa-dc96386e8a66-19ac3380-b87e-49ed-9736-82ad58c801f8-probas.csv',
                     #'earthquake/out/3afbbef8-6cfb-41b4-a8bc-0f6751c47301-9e05f9ba-fb2e-48b4-8d9f-ce70b6950a8a-probas.csv',]

        _, _ = combine_and_write(
            probas_fns,
            "earthquake/out",
            "building_id",
            "damage_grade",
            classes= ["1", "2", "3"],
            make_preds = True
        )



if __name__ == "__main__":
    main()
