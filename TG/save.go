package TG

import (
	"io/ioutil"
	"encoding/json"
	"fmt"
	"os"
)

type JSON_Tensor struct {
	Shape    string
	Data     string
	BoolData string
	Batched  string
}

// This function marshals the members of a tensor to JSON and returns a JSON_Tensor
// Note, the JSON_Tensor itself is not JSON, it is a struct with string members
func MarshalTensor(A *Tensor) *JSON_Tensor {

	// Marshal Tensor Members to JSON
	Data_JSON, err := json.Marshal(A.Data)
	if err != nil {
		panic(err)
	}
	BoolData_JSON, err := json.Marshal(A.BoolData)
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
		Shape:    string(Shape_JSON),
		Data:     string(Data_JSON),
		BoolData: string(BoolData_JSON),
		Batched:  string(Batched_JSON),
	}
	return result
}

// This function marshals an entire tensor to JSON and writes it to the specified fileName
func Save_JSON(fileName string, A *Tensor) {

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
