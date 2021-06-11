# DLAD Exercise 3

### AWS Setup

If not already done, please follow [aws/SETUP.md](aws/SETUP.md) to setup your AWS access.

### AWS Development and Testing

You can launch a development AWS EC2 CPU instance using:

```shell script
python3 aws_start_instance.py --mode devel
```

It'll have the requirements installed and dataset available. Your local code is uploaded during the initialization as
well.

You can attach to the tmux session of the instance using the last printed ssh command.
To test your source code for task 1, 2, 4, and 5, you can run following script on the EC2 instance:

```shell script
python tests/test.py --task X  # , where X is the task number.
```

As the instance has to warm up, the **first call will take several minutes** until it runs. If you want to 
update the source code on the AWS instance, you can run the rsync command printed by aws_start_instance.py. After you 
finished testing, please stop the instance using:

```shell script
bash aws/stop_self.sh
```

In case you forget to stop the instance, there is a 4 hour timeout. The development instance is only intended for 
task 1-5, which do no require a GPU. If you want to launch the tests automatically on the instance, refer to 
[aws/devel_in_tmux.sh](aws/devel_in_tmux.sh).

### AWS Training

You can launch a training on an AWS EC2 GPU instance using:

```shell script
python aws_start_instance.py --mode train
```

During the first run, the script will ask you for some information such as the wandb token for the setup.
You can attach to the launched tmux session by running the last printed command. If you want to close the connection
but keep the script running, detach from tmux using Ctrl+B and D. After that, you can exit the ssh connection, while
tmux and the training keep running. You can enter the scroll mode using Ctrl+B and [ and exit it with Q. 
In the scroll mode, you can scroll using the arrow keys or page up and down. Tmux has also some other nice features
such as multiple windows or panels (https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/). Please note
that there is a **timeout** of 24 hours to the instance. If you find that not sufficient, please adjust 
`TIMEOUT_TRAIN = 24  # in hours`
in [aws_start_instance.py](aws_start_instance.py). To check if you are unintentionally using AWS resources, you can
have a look at the AWS cost explorer: https://console.aws.amazon.com/cost-management/home?region=us-east-1#/dashboard.

You can change the training hyperparameters in [config.yaml](config.yaml). 

Please note that the first epoch will train considerably slower than the following ones as the required parts of the
AWS volume are downloaded from S3 on demand.

### AWS Interactive Development

During developing your own code, you'll often run into the problem that the training crashes briefly after the start due
to some typo. In order to avoid the overhead of waiting until AWS allows you to start a new instance as well as the
instance setup, you can continue using the same instance for further development. For that purpose cancel the automatic
termination using Ctrl+C. Fix the bug in your local environment and update your AWS files by running the rsync command, 
which was printed by aws_start_instance.py, on your local machine. After that, you can start the training on the AWS 
instance by running:
```shell script
cd ~/code/ && bash aws/train.sh
``` 

### Weights and Biases Monitoring

You can monitor the training via the wandb web interface https://wandb.ai/home. If you have lost the ec2 instance 
information for a particular (still running) experiment, you can view it by choosing the 
Table panel on the left side and horizontally scroll the columns until you find the EC2 columns. 
You can even use the web interface to stop a run (click on the three dots beside the run name and choose Stop Run). 
After you stopped the run, it'll still do the test predictions and terminate its instance afterwards. If you do not 
stop a run manually, it will terminate it's instance as well after completion.

In the workspace panel, we recommend switching the x-axis to epoch (x icon in the top right corner) for
visualization.
The logged histograms, you can only view if you click on a single run.

### AWS S3 Checkpoints and Submission Zip

To avoid exceeding the free wandb quota, the checkpoints and submission zips are saved to AWS S3. The link is logged
to wandb. You can find it on the dash board (https://wandb.ai/home) in the Table panel (available on the left side)
in the column S3_Link. 

Use the following command to download a submission archive to the local machine:

```shell script
aws s3 cp <s3_link> <local_destination>
```

### Resume Training

If a spot instance was preempted, you can resume a training by providing the trainer.resume_from_checkpoint flag 
in [config.yaml](config.yaml). To find out the checkpoint path, go to the wandb Table panel 
(available on the left side) and checkout the column S3_Path.
 
