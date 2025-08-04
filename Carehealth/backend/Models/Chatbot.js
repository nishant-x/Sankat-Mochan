const mongoose = require("mongoose");

const ChatbotSchema = new mongoose.Schema({
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: "Patient", required: true },
    prescription: [
        {
          medicine_name: { type: String, required: true, trim: true },
          quantity: { type: Number, required: true, min: 1 },
          dosage: { type: String, required: true, trim: true },
          duration: { type: String, trim: true }, // Optional: Duration of the medication
        },
    ],
    reportDetails: { type: String, required: true },
    department: {
        type: [String], // Array of strings
        default: ["Cardiology", "Neurology", "Orthopedics", "Pediatrics"] // Default values
    },
    createdAt: { type: Date, default: Date.now } 
});

module.exports = mongoose.model("Chatbot", ChatbotSchema);
