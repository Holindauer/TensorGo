# TensorGo 

-----------------------------------------------------------------------------------------------------
<center>
<h3>TensorGo is a minimal Tensor library for the Go Programming Language.</h3>
    
<p align="center">
  <img src="tensor_visualization.jpg" alt="Description" style="width:50%">- Neural Networks 
</p>


</center>

## Supported Features
- Operations on Arrays of n dimmensions
- Reverse Mode Autodiff for Backpropgation
- Linear Algebra Functionality
- Neural Networks (*limited*)


## Setup
-----------------------------------------------------------------------------------------------------

To set up this use this package in your own project, first install the package using the following command:

    go get github.com/Holindauer/TensorGo

Then import the package into your project:

    import . "github.com/Holindauer/TensorGo/TensorGo"

Don't forget the . before the import statement. This allows you to call the functions and methods in this package without having to specify the package name.

## Documentation
-----------------------------------------------------------------------------------------------------


The documentation for this library can be found here: [TG Doc](documentation.md).


## Contributing
-----------------------------------------------------------------------------------------------------


If you would like to contriibute to this poroject, take on an issue and submit a [pull request](pull_request_template.md). If you would like to add a new feature, please open an issue first to discuss the feature you would like to add. 

The two most import rules are: 

    1: All changes must pass testing 
    2: Reduce code duplication as much as possible


For more information on the contribution procedure and coding standards, please visit our [Contribution Guidelines](CONTRIBUTING.md) 

Also, take a look at the [Project Structure](project_structure.md) file to get a sense for how the project is organized: 

## Discord
-----------------------------------------------------------------------------------------------------
For more information, or to ask questions about functionality or contribution, join our Discord server: [TG Discord](https://discord.gg/mEy8F49Szu)

## Extra Thoughts, Reflections

This was my first project that involved major software engineering challenges. The overal goal going into the project was to understand Tensors better, collaborate with other developers, improve at software engineering by pushing at something really difficult.

 I think that given the time I started it, it has turned out better than I expected, but were I to go back and reimplement it there would be a lot of changes. This project was where I discovered the need for testing, when the project began I did not know much about how to test code and as such, only began writing tests midway through the project. Because of this, the Tests directory is not as thorough as I'd want it to be. Another issue I ran into that I had not experienced before was watching the codebase grow to the point of being unruly, my projects, up until starting this one were all realively self contained and minimal. Currently, TensorGo is abouy 1750 lines of code. I found myself with a bit too many features that were difficult to maintain and, as such, found myself refactoring often and the code being a bit messier than I like. 

Overall, I'm pretty happy with how this turned out and the features I was able to implement. Though much improvement is to be made wrt execution on the next.

    -------------------------------------------------------------------------------
    Language                     files          blank        comment           code
    -------------------------------------------------------------------------------
    Go                              16            734            975           1744
    -------------------------------------------------------------------------------
    SUM:                            16            734            975           1744
    -------------------------------------------------------------------------------