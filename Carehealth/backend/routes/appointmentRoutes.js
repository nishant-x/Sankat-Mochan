const express = require("express");
const Appointment= require("../Models/AppointmentSchema");
const router = express.Router();

router.get("/getAppointments/:id", (req, res) => {
    const id = req.params.id;

    Appointment.find({ patientId: id }) // ✅ Find all appointments for the patient
        .then((appointments) => {
            if (appointments.length === 0) {
                return res.status(404).json({ message: "No appointments found for this patient" });
            }
            res.json(appointments); // Send all appointments
        })
        .catch((err) => {
            res.status(500).json({ message: "Error fetching appointments", error: err.message });
        });
});


router.get("/total-appointment-count", async (req, res) => {
    try {
        const totalAppointments = await Appointment.countDocuments(); // ✅ Await the result
        res.json({ totalAppointments });
    } catch (error) {
        res.status(500).json({ message: "Error fetching total appointments", error });
    }
});


// Route to fetch today's appointments
router.get("/today", async (req, res) => {
    try {
        const today = new Date().toISOString().split("T")[0];
        const appointments = await Appointment.find({ date: today });
        res.json(appointments);
    } catch (error) {
        res.status(500).json({ message: "Error fetching appointments", error });
    }
});

//Route to show patient Total Appointment
router.get('/patient/total_appointment/:id', async(req,res)=>{
    try{
        const id=req.params.id;
        console.log(id)
        const total_appointment=await Appointment.countDocuments({patientId:id});
        res.json({total_appointment});
        console.log(total_appointment)

    }catch(error){
        res.status(500).json({ message: "Error fetching total appointments", error });
    }
})

// Route to show latest Appointment of patient
router.get('/patient/latest_appointment/:id', async (req, res) => {
    try {
        const id = req.params.id;
        console.log('new id from backend:', id );

        // Convert today's date to string in YYYY-MM-DD format
        const today = new Date().toISOString().split('T')[0];

        // Find the earliest appointment that is today or in the future
        const latestAppointment = await Appointment.findOne({
            patientId: id,
            date: { $gte: today }  // Compare as a string
        }).sort({ date: 1 }); // Sort in ascending order to get the closest future appointment
        
        console.log("latestAppointment:", latestAppointment);

        if (!latestAppointment) {
            return res.status(404).json({ message: "No upcoming appointments found." });
        }

        res.json(latestAppointment);
    } catch (error) {
        console.error("Error fetching latest appointment:", error);
        res.status(500).json({ message: "Error fetching latest appointment", error: error.message });
    }
});


module.exports=router;
