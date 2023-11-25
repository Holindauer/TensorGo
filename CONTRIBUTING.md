# Contribution Gudielines

The following is a list of guidelines for contributing to this project.

# General

## Pull Requests
When opening a pull request, fill out the [pull request template](pull_request_template.md) 

## Changes must pass all tests in the Tests directory
The Go-LinAlg/Tests directory contains tests for all functions in the library. Run "go test" in the Tests directory to run all tests. If you make a change that causes a test to fail, the change will not be accepted.

## Refactor Code to Reduce Duplication
Changes should be as simple as possible while still being correct. If you see a way to refactor code to reduce duplication, do it.

## Update the README
If you make a shcange that affects the way an operation is used, update the readme. 

# Style Guide

## Terse Code is Happy Code :)
Code in this library should be as terse as possible while still being readable. Ideal code is easy to read and self documenting while still being as short as possible. 

Continuous refactoring of code is appreciated. If you see a 5 functions that can be combined into 1, do it. As long as the Tests still pass, the change will be accepted.

Changes made to make code better adhere to the style guide are always welcome!

## Comments
Comments should be concise and to the point. For complex functions that are integral to the entire operation, such as Index(), longer explanations are acceptable, though they should be as consise as possible while still upholding clarity of communication. 

Comments should not be used in situations where the code is obvious. Unnecessary clutter will not be accepted.

## Refering to Tensors
When reffering to Tensor arguments inside of a function, the convention is to use A, then B,C,D and so on for subsequent arguments.

## All Python Code must be Typed
All Python code must be typed. This is to ensure that the Python code is as readable as possible.