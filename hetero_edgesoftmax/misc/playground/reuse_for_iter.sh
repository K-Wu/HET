#!/bin/bash

# Define a function containing the loops
my_loops() {
  for i in {1..5}; do
    $1
    for j in {1..3}; do
      $2
    done
  done
}

# Call the function with different statements as arguments
my_loops "echo \"Hello, world!\"" "echo \"Hello, kitty!\""