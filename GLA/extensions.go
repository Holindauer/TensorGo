package GLA

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// This source file contains functions related to retrieving and using extensions to the library. Extensions are defined as anything that
// must be downloaded from a remote repository in order to be used. Currently, the only extension is the Linear Systems Approximator.

//----------------------------------------------------------------------------------------------Linear Systems Approximator

func Get_LinSys_Approximator() {
	fmt.Println("Downloading LinSys_Approximator...")

	// Adjusted path to the script
	cmd := exec.Command("sh", "./Scripts/LinSys_Approximator/check_repository_download.sh")

	// Capture and display standard output and standard error
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		fmt.Println("Error: ", err)
	}
}

// This function trains the Linear Systems Approximator on a matrix of the specified type, size, and fill percentage.
func Train_LinSys_Approximator(matrixType string, aSize int, fillPercentage float64) error {
	fmt.Println("Training Linear Systems Approximator...")

	// The script path relative to the current working directory
	scriptPath := filepath.Join("Scripts", "LinSys_Approximator", "run_training.sh")

	// Convert aSize and fillPercentage to string
	aSizeStr := fmt.Sprintf("%d", aSize)
	fillPercentageStr := fmt.Sprintf("%f", fillPercentage)

	// Prepare the command to execute the script with arguments
	cmd := exec.Command("bash", scriptPath, matrixType, aSizeStr, fillPercentageStr)

	// Direct standard output and standard error to the respective os.Stdout and os.Stderr
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Run the command
	err := cmd.Run()
	if err != nil {
		fmt.Println("Error running training script:", err)
		return err
	}

	return nil
}
