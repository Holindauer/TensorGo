package GLA

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

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

// Train_LinSys_Approximator runs the run_training.sh Bash script with specified arguments
func Train_LinSys_Approximator(matrixType string, aSize int, fillPercentage float64) error {
	fmt.Println("Training Linear Systems Approximator...")

	// The script path relative to the current working directory
	scriptPath := filepath.Join("Scripts", "LinSys_Approximator", "run_training.sh")

	// Convert aSize and fillPercentage to string
	aSizeStr := fmt.Sprintf("%d", aSize)
	fillPercentageStr := fmt.Sprintf("%f", fillPercentage)

	// Prepare the command to execute the script with arguments
	cmd := exec.Command("bash", scriptPath, matrixType, aSizeStr, fillPercentageStr)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("error running training script: %v, output: %s", err, string(output))
	}

	fmt.Println("Training script output:", string(output))
	return nil
}
