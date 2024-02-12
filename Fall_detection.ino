#include <TensorFlowLite.h> 
#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "Arduino_BMI270_BMM150.h"
#include <ArduinoBLE.h>

BLEService tfliteService("12345678-1234-5678-1234-567812345678");  // Define a custom service UUID
BLECharacteristic tflitePrediction("87654321-1234-5678-1234-567812345678", BLERead | BLENotify, 10);  // Define a custom characteristic UUID

const int kInferencesPerCycle = 10;

const int numSamples = 10;
int samplesRead = numSamples;

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  int inference_count = 0;

  constexpr int kTensorArenaSize = 2000;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

const char* ACTIONS[] = {
  "ADL",
  "Fall"
};

#define NUM_ACTIONS (sizeof(ACTIONS) / sizeof(ACTIONS[0]))

void setup() {
  Serial.begin(9600);
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  model = tflite::GetModel(model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported version %d.",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  inference_count = 0;

  BLE.begin();
  BLE.setLocalName("TFLiteModel");
  BLE.setAdvertisedService(tfliteService);
  tfliteService.addCharacteristic(tflitePrediction);
  BLE.addService(tfliteService);
  tflitePrediction.writeValue("begin");  // Initialize the characteristic value

  // Start advertising the service
  BLE.advertise();
}

void loop() {

  float accx, accy, accz, gyx, gyy, gyz;

  // wait for significant motion
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      // read the acceleration data
      IMU.readAcceleration(accx, accy, accz);
      IMU.readGyroscope(gyx, gyy, gyz);
      samplesRead = 0;

      break;
    }
  }

  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples) {
    // check if new acceleration AND gyroscope data is available
    if (IMU.accelerationAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(accx, accy, accz);
      IMU.readGyroscope(gyx, gyy, gyz);

      input->data.f[0] = accx;
      input->data.f[1] = accy;
      input->data.f[2] = accz;
      input->data.f[3] = gyx;
      input->data.f[4] = gyy;
      input->data.f[5] = gyz;

      samplesRead++;

      if (samplesRead == numSamples) {
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          MicroPrintf("Invoke failed");
          return;
        }

        float fall_detection_output = output->data.f[0];
        //Serial.println(fall_detection_output, 4);
        if (fall_detection_output > 0.75) {
          tflitePrediction.writeValue("FALL");
          Serial.println("FALL!");
        }
      }
    }
  }
}