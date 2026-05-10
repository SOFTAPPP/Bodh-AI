from app.services.vector_service import VectorService
import os

if __name__ == "__main__":
    vs = VectorService()
    vs.clear_all_data()
    print("Vector store cleared. Please restart the app and re-upload the PDF to apply new chunking settings.")
