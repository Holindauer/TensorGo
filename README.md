# Tensor-Go 





-----------------------------------------------------------------------------------------------------
<center>
<h3>Tensor-Go is an open-source numerical computation and linear algebra library for the Go Programming Language.</h3>
    
<p align="center">
  <img src="tensor_visualization.jpg" alt="Description" style="width:50%">
</p>

<h3>TG is designed for easy use and efficiency when working with arrays of arbitrary dimmension.</h3>
</center>


Try out a Batched Matrix Multiplication:

    A := Range_Tensor([]int{4, 5, 5}, true) // batch of 4 tensors of shape [5, 5]
    AxA := MatMul(A, A, true)      
# Setup
-----------------------------------------------------------------------------------------------------

To set up this use this package in your own project, first install the package using the following command:

    go get github.com/Holindauer/Tensor-Go

Then import the package into your project:

    import . "github.com/Holindauer/Tensor-Go/TG"

Don't forget the . before the import statement. This allows you to call the functions and methods in this package without having to specify the package name.

# Documentation
-----------------------------------------------------------------------------------------------------


The documentation for this library can be found here: [TG Doc](documentation.md).


# Contributing
-----------------------------------------------------------------------------------------------------


If you would like to contriibute to this poroject, take on an issue and submit a [pull request](pull_request_template.md). If you would like to add a new feature, please open an issue first to discuss the feature you would like to add. 

The two most import rules are: 

    1: All changes must pass testing 
    2: Reduce code duplication as much as possible


For more information on the contribution procedure and coding standards, please visit our [Contribution Guidelines](CONTRIBUTING.md) 

Also, take a look at the [Project Structure](project_structure.md) file to get a sense for how the project is organized: 

# Discord
-----------------------------------------------------------------------------------------------------
For more information, or to ask questions about functionality or contribution, join our Discord server: [TG Discord](https://discord.gg/mEy8F49Szu)


