const express = require("express");
const Chatbot = require("../Models/Chatbot"); // Import Chatbot Schema
const PatientPrescriptions = require("../Models/Patientprescription"); // Import PatientPrescription Schema
const router = express.Router();

// Route to save medical report details along with prescription
router.post("/save-report", async (req, res) => {
    try {
      const { patientId, reportDetails } = req.body;
  
      if (!patientId || !reportDetails) {
        return res.status(400).json({ error: "All fields are required." });
      }
  
      // 🔹 Find the patient's prescription data
      const prescriptionData = await PatientPrescriptions.findOne({ patient_id: patientId });
  
      if (!prescriptionData) {
        return res.status(404).json({ error: "No prescription found for this patient." });
      }
  
      // 🔹 Ensure prescriptions exist
      const prescriptionList = Array.isArray(prescriptionData.prescription)
        ? prescriptionData.prescription.map((item) => ({
            medicine_name: item.medicine_name,
            quantity: item.quantity,
            dosage: item.dosage,
            duration: item.duration || "", // Optional
          }))
        : [];
  
      // 🔹 Create new report entry
      const newReport = new Chatbot({
        patientId,
        prescription: prescriptionList, 
        reportDetails,
      });
  
      // 🔹 Save to MongoDB
      const savedReport = await newReport.save();
      console.log("Saved Report:", savedReport);
  
      res.status(201).json({ message: "Report saved successfully!", report: savedReport });
  
    } catch (error) {
      console.error("Error saving report:", error);
      res.status(500).json({ error: "Failed to save report." });
    }
  });
  

module.exports = router;
