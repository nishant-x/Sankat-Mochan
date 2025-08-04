const express = require("express");
const router = express.Router();
const mongoose = require('mongoose');
const Appointment = require('../Models/AppointmentSchema');
const Patient = require('../Models/PatientRegistration');
const Doctor = require('../Models/DoctorRegistration');

router.post('/check-availability', async (req, res) => {
    try {
        const { date, department } = req.body;

        // Find all appointments for the given date and department
        const existingAppointments = await Appointment.find({ date, department });

        // Get all booked time slots
        const bookedSlots = existingAppointments.map(appointment => appointment.time);
        // console.log("bookedSlots:",bookedSlots,"existingAppointments:",existingAppointments);

        // Send back the booked slots as unavailable
        res.json({ unavailableSlots: bookedSlots });
    } catch (error) {
        console.error('Error checking availability:', error);
        res.status(500).json({ message: 'Error checking availability' });
    }
});


router.post("/book-appointment", async (req, res) => {
    try {
        const { name, patientId, email, phone, date, time, type, department } = req.body;

        // Validate required fields
        if (!name || !patientId || !email || !phone || !date || !time || !type || !department) {
            return res.status(400).json({ message: "All fields are required" }); 
        }

        // Find all doctors in the requested department
        const availableDoctors = await Doctor.find({ specialization: department });

        if (!availableDoctors.length) {
            return res.status(404).json({ message: "No doctors available in this department" });
        }

        // Find the doctor with the least appointments for load balancing
        let assignedDoctor = availableDoctors[0]; // Default to the first doctor

        for (let doctor of availableDoctors) {
            const count = await Appointment.countDocuments({ doctorId: doctor._id, date });
            if (count < (await Appointment.countDocuments({ doctorId: assignedDoctor._id, date }))) {
                assignedDoctor = doctor;
            }
        }

        // Check if the selected doctor's time slot is already booked
        const existingAppointment = await Appointment.findOne({
            date,
            time,
            doctorId: assignedDoctor._id,
        });

        if (existingAppointment) {
            return res.status(400).json({ message: "Selected time slot already booked for this doctor" });
        }

        // Create appointment
        const newAppointment = new Appointment({
            name,
            patientId,
            email,
            phone,
            date,
            time,
            type,
            department,
            doctorId: assignedDoctor._id, // Assign doctor ID
            doctorName: assignedDoctor.name, // Assign doctor name
            status: "Booked",
        });

        await newAppointment.save();

        res.status(201).json({
            message: "Appointment booked successfully",
            appointment: newAppointment
        });

    } catch (error) {
        console.error("Error booking appointment:", error);
        res.status(500).json({ message: "Internal server error" });
    }
});



// Endpoint to cancel an appointment
router.delete('/cancel-appointment/:id', async (req, res) => {
    try {
        const { id } = req.params;

        // Check if ID is a valid MongoDB ObjectId
        if (!mongoose.Types.ObjectId.isValid(id)) {
            return res.status(400).json({ message: 'Invalid appointment ID' });
        }

        // Delete the appointment
        const deletedAppointment = await Appointment.findByIdAndDelete(id);

        if (!deletedAppointment) {
            return res.status(404).json({ message: 'Appointment not found' });
        }

        res.json({ message: 'Appointment cancelled successfully' });

    } catch (error) {
        console.error('Error cancelling appointment:', error.stack);
        res.status(500).json({ message: 'Error cancelling appointment', error: error.message });
    }
});


// Endpoint to reschedule an appointment
router.put('/reschedule-appointment/:id', async (req, res) => {
    try {
        const { id } = req.params;
        const { date, time, department } = req.body;

        // Check if the new slot is available
        const existingAppointment = await Appointment.findOne({ date, time, department });
        if (existingAppointment) {
            return res.status(400).json({ message: 'New time slot already booked' });
        }

        // Update the appointment
        const updatedAppointment = await Appointment.findByIdAndUpdate(id, { date, time }, { new: true });

        if (!updatedAppointment) {
            return res.status(404).json({ message: 'Appointment not found' });
        }

        res.json({ message: 'Appointment rescheduled successfully', appointment: updatedAppointment });
    } catch (error) {
        console.error('Error rescheduling appointment:', error);
        res.status(500).json({ message: 'Error rescheduling appointment' });
    }
});

// Endpoint to get all appointments
router.get('/appointments', async (req, res) => {
    try {
        const appointments = await Appointment.find().sort({ date: 1, time: 1 });
        res.json(appointments);
    } catch (err) {
        res.status(500).json({ message: 'Error fetching appointments' });
    }
});
router.get('/appointments/:id', async (req, res) => {
    try {
        const { id } = req.params; // Extract patient ID from route parameters
        
        // Get today's date in "YYYY-MM-DD" format
        const currentDate = new Date().toISOString().split('T')[0];

        // Find all appointments from today onwards
        const patientAppointments = await Appointment.find({ 
            patientId: id,
            date: { $gte: currentDate } // Fetch today's and all future appointments
        }).sort({ date: 1, time: 1 });

        if (!patientAppointments.length) {
            return res.status(404).json({ message: 'No upcoming appointments found for this patient' });
        }

        res.json(patientAppointments); // Send filtered appointments
    } catch (err) {
        console.error('Error fetching appointments:', err);
        res.status(500).json({ message: 'Error fetching appointments' });
    }
});




module.exports = router;