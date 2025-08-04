const express = require("express");
const router = express.Router();
const PatientPrescriptions = require('../Models/Patientprescription');


// Route to create a patient percription
// POST route to add a patient's prescription
router.post('/save-prescription', async (req, res) => {
    const {
        patientName,
        doctorName,
        medicines,
        patientId,
        doctorId,
        handledByPharmacist,
        grandTotal,
    } = req.body;

    try {
        // Create a new patient prescription document
        const newPatient = new PatientPrescriptions({
            patient_name: patientName,
            doctor_name: doctorName,
            prescription: medicines.map((med) => ({
                medicine_name: med.name,
                quantity: med.quantity,
                dosage: med.dose,
                duration: med.duration,
            })),
            doctor_id: doctorId,
            patient_id: patientId,
            handled_by_pharmacist: handledByPharmacist || false,
            grand_total: grandTotal || 0,
        });

        // Save the document to the database
        const savedPatient = await newPatient.save();

        res.status(201).json({
            message: 'Prescription added successfully!',
            data: savedPatient,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            message: 'An error occurred while saving the prescription.',
            error: error.message,
        });
    }
});

// ###############################  patient section ###############################
// Route to show patient prescription 
router.get('/patient/prescription/:id', async (req, res) => {
    try {
        const patientId = req.params.id;
        console.log(patientId);
        const patientPrescriptions = await PatientPrescriptions.find({ patient_id: patientId });
        console.log(patientPrescriptions);

        if (!patientPrescriptions) {
            return res.status(404).json({ message: 'Prescription not found' });
        }
        res.json(patientPrescriptions);
    } catch (err) {
        console.error(err);
    }
});

router.get('/patient/latest-prescription/:id', async (req, res) => {
    try {
        const patientId = req.params.id;
        // console.log(patientId);

        const latestPrescription = await PatientPrescriptions.findOne({ patient_id: patientId }) 
        .sort({ prescription_date: -1 })
        .limit(2);
    

        // console.log("From Backend",latestPrescription);

        if (!latestPrescription) {
            return res.status(404).json({ message: 'Latest prescription not found' });
        }

        res.json(latestPrescription);
    } catch (err) {
        console.error('Error fetching latest prescription:', err);
        res.status(500).json({ message: 'Error fetching latest prescription', error: err.message });
    }
})

 


// router.get("save-file/patient/prescription/:id", async (req, res) => {
//     try {
//       const patientId = req.params.id;
//       console.log(`Fetching prescriptions for patient ID: ${patientId}`);
  
//       const patientPrescriptions = await PatientPrescriptions.find({ patient_id: patientId });
  
//       if (!patientPrescriptions || patientPrescriptions.length === 0) {
//         return res.status(404).json({ message: "No prescription found" });
//       }
  
//       res.json({ prescriptions: patientPrescriptions, patientId });
//     } catch (err) {
//       console.error(err);
//       res.status(500).json({ message: "Server Error" });
//     }
//   });
  

// Route to show the patient bills of parmacy
router.get('/patient/bill/:id', async (req, res) => {
    try {
        const patientId = req.params.id;

        // Fetch prescriptions for the given patient_id
        const patientPrescriptions = await PatientPrescriptions.find({ 
            patient_id: patientId, 
            handled_by_pharmacist: true 
        });
        

        if (patientPrescriptions.length === 0) {
            // Send a 404 response if no prescriptions are found
            return res.status(404).json({ message: 'No prescriptions found for this patient' });
        }

        // Send the data to the client
        res.status(200).json({ patientPrescriptions });
    } catch (err) {
        // Handle any errors
        console.error('Error fetching prescriptions:', err.message);
        res.status(500).json({ message: 'Failed to fetch prescriptions', error: err.message });
    }
});

module.exports = router;
