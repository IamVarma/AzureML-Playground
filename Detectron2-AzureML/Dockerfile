FROM openvino/onnxruntime_ep_ubuntu18

WORKDIR /app

USER root

#RUN apt-get update && apt-get install -y curl

RUN curl "https://edgemini.blob.core.windows.net/models/model-buffle.onnx?sv=2020-02-10&st=2023-03-21T03%3A24%3A08Z&se=2023-04-30T04%3A24%3A00Z&sr=b&sp=racwdmeop&sig=Ilyxz22mjO%2BG45nfgUbqNAwMdwTIVY%2B4f4vC4Po2uV4%3D" --output "model-buffle.onnx"
