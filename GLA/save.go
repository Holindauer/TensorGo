package GLA

import (
//    "io/ioutil"
    "os"
    "encoding/json"
    "strconv"
    "reflect"
    "strings"
)

type JSON_Tensor struct {
    Shape    string
    Data     string
    BoolData string
    Batched  string
}

func SliceToString(slice interface{}, converter func(interface{}) string) string {
	values := make([]string, 0)

	sliceValue := reflect.ValueOf(slice)
	for i := 0; i < sliceValue.Len(); i++ {
		element := sliceValue.Index(i).Interface()
		values = append(values, converter(element))
	}
	return strings.Join(values, ",")
}

func MarshalTensor(tensor *Tensor) (JSON_Tensor) {    
	intResult := SliceToString(tensor.Shape, func(v interface{}) string {
		return strconv.Itoa(v.(int))
	})

	floatResult := SliceToString(tensor.Data, func(v interface{}) string {
		return strconv.FormatFloat(v.(float64), 'f', -1, 64)
	})
    
	boolResult := SliceToString(tensor.BoolData, func(v interface{}) string {
		return strconv.FormatBool(v.(bool))
	})

    result := JSON_Tensor {
        Shape: intResult,
        Data: floatResult,
        BoolData: boolResult,
        Batched: strconv.FormatBool(tensor.Batched),
    }
    return result
}

func Save(filename string, json_tensor *JSON_Tensor) {
    jsonData, err := json.Marshal(json_tensor)
    if err != nil {
        panic(err)
    }

    file, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
    if err != nil {
        panic(err)
    }
    defer file.Close()

    if _, err := file.Write(jsonData); err != nil {
        panic(err)
    }
}
