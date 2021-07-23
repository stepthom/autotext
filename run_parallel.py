import argparse
import subprocess
import tempfile


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run-settings', default='default_settings.json')
    parser.add_argument('-p', '--num-pumps', type=int, default=0)
    parser.add_argument('-e', '--num-earthquakes', type=int, default=0)
    parser.add_argument('-n', '--num-h1n1s', type=int, default=0)
    parser.add_argument('-s', '--num-seasonals', type=int, default=0)
    args = parser.parse_args()
    
    c_types = ["pump" for i in range(args.num_pumps)]
    c_types = c_types + ["earthquake" for i in range(args.num_earthquakes)]
    c_types = c_types + ["h1n1" for i in range(args.num_h1n1s)]
    c_types = c_types + ["seasonal" for i in range(args.num_seasonals)]
    
    print(c_types)
   
    i = 0
    processes = []
    for c_type in c_types:
        i = i+1
        print("run_parallel: Running search on {}.".format(c_type))
        
        stdout_f = tempfile.NamedTemporaryFile(prefix="{}-{}-stdout-".format(c_type, i), dir="logs", suffix=".log", delete=False)
        stderr_f = tempfile.NamedTemporaryFile(prefix="{}-{}-stderr-".format(c_type, i), dir="logs", suffix=".log", delete=False)
        print("run_parallel: stdout going to file {}".format(stdout_f.name))
        print("run_parallel: stderr going to file {}".format(stderr_f.name))
        cmd = [
            "python", "run_seq.py", 
            "--competition", c_type, 
            "--num-to-run", "5", 
            "--run-settings", args.run_settings,
        ] 
        p = subprocess.Popen(cmd,
                         stdout=stdout_f,
                         stderr=stderr_f,
                         universal_newlines=True,
                         bufsize=1,
                        )
        processes.append(p)
    
    print("run_parallel: Waiting on {} processes:".format(len(processes)))
    for p in processes:
        print("run_parallel: p.pid={}".format(p.pid))
    exit_codes = [p.wait() for p in processes]
    print("exit codes: {}".format(exit_codes))
        
if __name__ == "__main__":
    main()
