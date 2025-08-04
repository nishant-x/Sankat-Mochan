const express = require("express");
const router = express.Router();
const Doctor = require('../Models/DoctorRegistration');
const Appointment = require('../Models/AppointmentSchema');


router.get('/doctor_patient_list/:id', async (req, res) => {
    try {
        const id = req.params.id;
        const doctor = await Doctor.findById(id);

        if (!doctor) {
            return res.status(404).json({ message: "Doctor not found" });
        }

        const department = doctor.specialization;
        const todayDate = new Date().toISOString().split("T")[0]; // Get today's date in "YYYY-MM-DD" format
        const patients = await Appointment.find({ department: department, date: todayDate });
        
        console.log("doctor data:", doctor, "patients:", patients);

        // Corrected sort function
        res.status(200).json({
            doctor,
            patients: patients.sort((a, b) => new Date(a.date) - new Date(b.date))
        });

    } catch (error) {
        console.error("Error fetching doctor or patients:", error);
        res.status(500).json({ message: "Internal Server Error" });
    }
});

// Example backend route (Node.js/Express)
router.get("/doctor/:doctor_id", async (req, res) => {
    try {
      const doctor = await Doctor.findById(req.params.doctor_id);
      if (!doctor) {
        return res.status(404).json({ error: "Doctor not found" });
      }
      res.json({ doctor });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: "Server error" });
    }
  });
// Update the reached if the patient prescription is done
router.put('/doctor_patient_list', async (req, res) => {
    try {
        const { patientId, reached } = req.body;

        if (!patientId) {
            return res.status(400).json({ message: "Patient ID is required" });
        }

        // Find and update the application
        const updatedApplication = await Appointment.findOneAndUpdate(
            { _id: patientId },
            { reached: Boolean(reached) }, // Convert to Boolean
            { new: true }
        );

        if (!updatedApplication) {
            return res.status(404).json({ message: "No matching application found" });
        }

        res.json({
            message: "Patient reached status updated successfully",
            updatedApplication,
        });
    } catch (error) {
        console.error("Error updating reached status:", error);
        res.status(500).json({ message: "Error updating reached status" });
    }
});

module.exports = router;



module.exports = router;