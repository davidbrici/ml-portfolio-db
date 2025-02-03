# Basic Bash and Bash Scripting Guide for ML Engineers

## Introduction to Bash
**Bash (Bourne Again Shell)** is a command-line interface and scripting language used for **automation, file handling, and environment management**, making it essential for **ML engineers** working with Linux and cloud environments.

---
## 1. Basic Bash Commands
### Navigating the Filesystem
```bash
pwd   # Print current working directory
ls    # List files in the directory
cd    # Change directory
cd .. # Move up one directory
```

### File Operations
```bash
touch file.txt  # Create an empty file
mkdir new_dir   # Create a new directory
rm file.txt     # Remove a file
rm -r new_dir   # Remove a directory and its contents
mv file1 file2  # Rename or move a file
cp file1 file2  # Copy a file
```

### Viewing File Contents
```bash
cat file.txt    # Display entire file
less file.txt   # View file with pagination
head -n 5 file.txt  # Show first 5 lines
tail -n 5 file.txt  # Show last 5 lines
grep "word" file.txt  # Search for a word in a file
```

---
## 2. Variables and Environment
### Defining Variables
```bash
name="ML Engineer"
echo "Hello, $name!"
```

### Environment Variables
```bash
echo $HOME  # Show home directory
echo $PATH  # Show system path
export VAR="value"  # Set a temporary environment variable
```

---
## 3. Writing Bash Scripts
### Creating a Script
1. Open a file: `nano script.sh`
2. Add the following content:
```bash
#!/bin/bash
# Simple Bash script
echo "Hello, Machine Learning Engineer!"
```
3. Make it executable:
```bash
chmod +x script.sh
```
4. Run the script:
```bash
./script.sh
```

---
## 4. Conditional Statements
### If-Else Example
```bash
#!/bin/bash
num=10
if [ $num -gt 5 ]; then
    echo "Number is greater than 5"
else
    echo "Number is 5 or less"
fi
```

---
## 5. Loops
### For Loop
```bash
#!/bin/bash
for i in {1..5}; do
    echo "Iteration $i"
done
```

### While Loop
```bash
#!/bin/bash
count=1
while [ $count -le 5 ]; do
    echo "Count: $count"
    ((count++))
done
```

---
## 6. Functions
```bash
#!/bin/bash
function greet() {
    echo "Hello, $1!"
}
greet "ML Engineer"
```

---
## 7. Working with Files and Data
### Reading a File Line by Line
```bash
#!/bin/bash
while read line; do
    echo "Line: $line"
done < file.txt
```

### Redirecting Output
```bash
ls > output.txt  # Save output to a file
echo "Log entry" >> log.txt  # Append output
```

### Finding and Replacing in Files
```bash
sed -i 's/old_text/new_text/g' file.txt  # Replace text in a file
```

---
## 8. Process Management
### Running and Monitoring Processes
```bash
ps aux   # List running processes
kill PID  # Kill a process by PID
htop     # Interactive process viewer
```

### Running Commands in the Background
```bash
python train_model.py &  # Run process in the background
jobs  # List background jobs
```

---
## 9. Scheduling Tasks with Cron Jobs
### Opening Cron Editor
```bash
crontab -e
```

### Scheduling a Job (Runs Every Day at Midnight)
```bash
0 0 * * * /path/to/script.sh
```

---
## 10. Working with SSH and Remote Machines
### Connecting to a Remote Server
```bash
ssh user@remote-server
```

### Copying Files Securely
```bash
scp local_file user@remote-server:/remote/path
```

---
## Conclusion
Bash scripting is essential for **automation, cloud computing, and ML workflows**. Mastering these basics will make you **more efficient in managing environments and workflows**.

For more advanced topics, check out guides on **Docker, Kubernetes, and MLOps**!

Happy scripting! 🚀