package TG

/*
* IO.go contains functions for loading and saving Tensors to and from files
 */

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strconv"
)

//===================================================================================================================== JSON IO

type JSON_Tensor struct {
	Shape    string
	Data     string
	BoolData string
	Batched  string
}

/*
* @notice marshals the members of a tensor to JSON and returns a JSON_Tensor
* Note, the JSON_Tensor itself is not JSON, it is a struct with string members
 */
func MarshalTensor(A *Tensor) *JSON_Tensor {

	// Marshal Tensor Members to JSON
	Data_JSON, err := json.Marshal(A.Data)
	if err != nil {
		panic(err)
	}
	Shape_JSON, err := json.Marshal(A.Shape)
	if err != nil {
		panic(err)
	}
	Batched_JSON, err := json.Marshal(A.Batched)
	if err != nil {
		panic(err)
	}

	result := &JSON_Tensor{
		Shape:   string(Shape_JSON),
		Data:    string(Data_JSON),
		Batched: string(Batched_JSON),
	}
	return result
}

// This function marshals an entire tensor to JSON and writes it to the specified fileName
func (A *Tensor) Save_JSON(fileName string) {

	// Marshal the Tensor
	A_JSON, err := json.Marshal(A)
	{
		if err != nil {
			panic(err)
		}
	}

	// Create and open the file
	file, err := os.Create(fileName)
	if err != nil {
		fmt.Println("Error creating file:", err)
		os.Exit(1)
	}
	defer file.Close()

	// Write something to the file
	_, err = file.WriteString(string(A_JSON))
	if err != nil {
		// Handle the error
		fmt.Println("Error writing to file:", err)
		os.Exit(1)
	}

	// Success message
	fmt.Println(fileName, " written successfully")
}

/*
* @notice This function loads a tensor from a JSON file and returns it
 */
func Load_JSON(fileName string) *Tensor {
	file, err := os.Open(fileName)
	if err != nil {
		fmt.Println("Error open file:", err)
		os.Exit(1)
	}
	defer file.Close()

	jsonData, err := ioutil.ReadAll(file)
	if err != nil {
		fmt.Println("Error parse json data")
		os.Exit(1)
	}

	var A *Tensor
	err = json.Unmarshal(jsonData, &A)
	if err != nil {
		fmt.Println("Error unmarshal json", err)
		os.Exit(1)
	}

	fmt.Println(fileName, " loaded successfully")
	return A
}

//===================================================================================================================== CSV IO

/*
* @notice LoadCSV() loads a CSV file into a Tensor
* @dev Data loaded from a csv is loaded as a batch of vectors
 */
func LoadCSV(csvFile string, skipHeader bool) *Tensor {

	// Print cwd
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get current working directory: %s", err)
	}
	fmt.Println("Current working directory: ", cwd)

	// Open the CSV file
	file, err := os.Open(csvFile)
	if err != nil {
		log.Fatalf("Failed to open CSV file: %s", err)
	}
	defer file.Close()

	// Read the CSV file
	reader := csv.NewReader(bufio.NewReader(file))
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Failed to read CSV file: %s", err)
	}

	// Create a Tensor to store the data
	irisTensor := Zero_Tensor([]int{len(rawCSVData), len(rawCSVData[0])}, true)

	// Convert CSV data into Tensor
	for i, row := range rawCSVData {

		// skip header if specified
		if i == 0 && skipHeader {
			continue
		}

		for j, val := range row {
			// Parse the string value into a float
			floatVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatalf("Failed to convert string to float: %s", err)
			}

			irisTensor.Data[irisTensor.Index([]int{i, j})] = floatVal
		}
	}

	return irisTensor
}

/*
* @notice SaveCSV() saves a Tensor to a CSV file
* @dev Tensors are saved as a batch of vectors. If data is multi-dimensional, it will be flattened.
 */
func SaveCSV(A *Tensor, fileName string) {
	// Create and open the file
	file, err := os.Create(fileName)
	if err != nil {
		fmt.Println("Error creating file:", err)
		os.Exit(1)
	}
	defer file.Close()

	// Write the data to the file
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write each row of the tensor to the CSV file
	for i := 0; i < A.Shape[0]; i++ {
		var row []string
		for j := 0; j < A.Shape[1]; j++ {
			// Calculate the index for the current element
			index := i*A.Shape[1] + j // Assuming a 2D tensor
			row = append(row, strconv.FormatFloat(A.Data[index], 'f', -1, 64))
		}

		// Write the row to the CSV file
		if err := writer.Write(row); err != nil {
			fmt.Println("Error writing to file:", err)
			os.Exit(1)
		}
	}

	// Check for errors on flush
	if err := writer.Error(); err != nil {
		fmt.Println("Error flushing file:", err)
		os.Exit(1)
	}

	// Success message
	fmt.Println(fileName, " written successfully")
}
