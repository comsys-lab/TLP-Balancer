#!/usr/bin/python3
# Usage: ./parsing.py results/tlp_balancer.txt
# Prints average queuing delay (2nd column) and average inference time (3rd column).
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'results/tlp_balancer.txt'

with open(path, 'r') as file:
    lines = file.readlines()

sum_col2 = 0.0
sum_col3 = 0.0
count = 0

for line in lines:
    if line[0] == '-':
        break
    parts = line.split()
    if len(parts) >= 3:
        sum_col2 += float(parts[1])
        sum_col3 += float(parts[2])
        count += 1

average_col2 = sum_col2 / count if count > 0 else 0
average_col3 = sum_col3 / count if count > 0 else 0

print(f'Average of second column: {average_col2}')
print(f'Average of third column: {average_col3}')
