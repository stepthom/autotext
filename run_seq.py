import argparse
import subprocess

def main():    
    parser = argparse.ArgumentParser(description="Calls run_one.py N times sequentially.")
    parser.add_argument('-s', '--run-settings', default='default_settings.json')
    parser.add_argument('-c', '--competition', default='pump')
    parser.add_argument('-n', '--num-to-run', type=int, default=10)
    args = parser.parse_args()
    
    for i in range(args.num_to_run):
        print("run_seq: Running {} of {}".format(i+1, args.num_to_run))
        
        cmd = [
            "python", "run_one.py", 
            "--run-settings", args.run_settings,
            "--competition", args.competition
        ] 
        
        p = subprocess.Popen(cmd, universal_newlines=True, bufsize=1)
        exit_code = p.wait()
        print("run_seq: Process exited with code {}".format(exit_code))
    
if __name__ == "__main__":
    main()
