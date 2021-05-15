//
//  Coordinator.swift
//  Smartcam
//
//  Created by venu vasudevan on 4/24/21.
//

import SwiftUI

import CoreML
import Vision

class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    
    @Binding var isCoordinatorShown: Bool
    @Binding var imageInCoordinator: Image?
    @Binding var labelFromCoordinator: String
    
    // Define a Vision classification request
    private lazy var classificationRequest: VNCoreMLRequest = {
      do {
        // create an instance of a pre-trained ML Model (Resnet in this case
        let model = try VNCoreMLModel(for: Resnet50Int8LUT().model)
        
        
        // create an image analysis request object with a completion handler that renders the most likely
        // label as an overlay on the image
        let request = VNCoreMLRequest(model: model) { request, _ in
            if let classifications =
              request.results as? [VNClassificationObservation] {
           
              let topClassifications = classifications.prefix(2).map {
                  (confidence: $0.confidence, identifier: $0.identifier)
              }
             
              
                self.labelFromCoordinator = topClassifications[0].identifier
            }
        }
        
        // Use Vision to crop the input image to match what the model expects
        request.imageCropAndScaleOption = .centerCrop
        
        return request
        
      } catch {
        // 5
        fatalError("Failed to load Vision ML model: \(error)")
      }
    }()

    func classifyImage(_ image: UIImage) {
      // 1
      guard let orientation = CGImagePropertyOrientation(
        rawValue: UInt32(image.imageOrientation.rawValue)) else {
        return
      }
      guard let ciImage = CIImage(image: image) else {
        fatalError("Unable to create \(CIImage.self) from \(image).")
      }
      // 2
      DispatchQueue.global(qos: .userInitiated).async {
        let handler =
          VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
        do {
          try handler.perform([self.classificationRequest])
        } catch {
          print("Failed to perform classification.\n\(error.localizedDescription)")
        }
      }
    }

    
    init(isShown: Binding<Bool>, image: Binding<Image?>, label: Binding<String>) {
        _isCoordinatorShown = isShown
        _imageInCoordinator = image
        _labelFromCoordinator = label
    }
    
    func imagePickerController(_ picker: UIImagePickerController,
                               didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        guard let unwrapImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage else { return }
        print("CoreML model call simulated")
        imageInCoordinator = Image(uiImage: unwrapImage)
        classifyImage(unwrapImage)
        
        isCoordinatorShown = false
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        print("cancelng coordinator")
        isCoordinatorShown = false
    }
}
