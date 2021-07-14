import argparse
import subprocess


def main():    
    parser = argparse.ArgumentParser()
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
        print("Running search on {}.".format(c_type))
        
        stdout_f = open("logs/{}-{}.stdout".format(c_type, i), "w")
        stderr_f = open("logs/{}-{}.stderr".format(c_type, i), "w")
        p = subprocess.Popen(["python", "search.py", 
                          "-c", c_type, 
                          "--run-at-most", "10", 
                          "--search-time", "8000"], 
                         stdout=stdout_f,
                         stderr=stderr_f,
                         universal_newlines=True,
                        )
        processes.append(p)
        
    exit_codes = [p.wait() for p in processes]
    print("exit codes: {}".format(exit_codes))
        
if __name__ == "__main__":
    main()
