//
//  ContentView.swift
//  ImageML
//
//  Created by venu vasudevan on 5/10/21.
//

import SwiftUI



struct CaptureImageView {
    @Binding var isShown: Bool
    @Binding var image: Image?
    @Binding var label: String
    
    func makeCoordinator() -> Coordinator {
        
        return Coordinator(isShown: $isShown, image: $image, label: $label)
    }
}

extension CaptureImageView : UIViewControllerRepresentable {
    
    func makeUIViewController(context: UIViewControllerRepresentableContext<CaptureImageView>) ->  UIImagePickerController {
        
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        //.camera for real-time, .photoLibrary for picking from gallery
        //picker.sourceType = .camera
        picker.sourceType = .photoLibrary
        //print("creating CaptureImageView")
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: UIViewControllerRepresentableContext<CaptureImageView>) {
        //print("updating CaptureImageView")
    }
}

struct ContentView: View {
    @State var image: Image? = nil
    @State var showCaptureImageView: Bool = false
    @State var label: String  = "TBD"
    var body: some View {
        ZStack {
            VStack {
                Button(action: {
                //print("button press")
                //print("capture image view --", self.showCaptureImageView)
                  self.showCaptureImageView.toggle()
                  
                  /// TODO 1: Add the action here
                }) {
                    Text("Choose photos")
                }
            
            image?.resizable()
              .frame(width: 250, height: 250)
              .clipShape(Circle())
              .overlay(Circle().stroke(Color.white, lineWidth: 4))
              .shadow(radius: 10)
                
            Text("Top label: \(label)") //this is the top label assigned by the ML algorithm (in this case ResNet
           }
        if (showCaptureImageView) {
          CaptureImageView(isShown: $showCaptureImageView, image: $image, label: $label)
        }
     }
  }
}


