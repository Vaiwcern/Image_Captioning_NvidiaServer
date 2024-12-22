## ACTIVATE ENVIRONMENT: 
	source ../env1/bin/activate

## FRAMEWORK: 
### Run step 1: 
	nohup python3 /home/ltnghia02/vischronos/scripts/first.py /home/ltnghia02/dataset /home/ltnghia02/vischronos/prompts/5 --begin_index 13692 --end_index 20000 --img_index 1 > /home/ltnghia02/vischronos/logs/step1_13692_20000_1.txt 2>&1 &
### Run step 234: 	
	nohup python3 /home/ltnghia02/vischronos/scripts/second.py /home/ltnghia02/dataset/ /home/ltnghia02/vischronos/prompts/4 --begin_index 4238 --end_index 5000 --img_index 1 > /home/ltnghia02/vischronos/logs/step234_4238_5000_1.txt 2>&1 &

### Run step 1 by .sh file: 
    ./scripts/first.sh <begin_index> <end_index> <image_index> <GPU>
	./scripts/first.sh 4238 5000 1 12
### Run step 2 by .sh file: 
    ./scripts/second.sh <begin_index> <end_index> <image_index> <GPU>
	./scripts/second.sh 4238 5000 1 3
	
## CHECK JOB RUNNING: 
### Step 1:
	ps aux | grep first.py
### Step 234:
	ps aux | grep second.py

## GPU USAGE: 
	nvitop

## COPY FILE TO SERVER: 
	scp <path_to_file> <auto_login_username>@ssh.axisapps.io:/home/ltnghia02