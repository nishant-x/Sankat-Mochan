const mongoose = require('mongoose');

const appointmentSchema = new mongoose.Schema({
    name: { type: String, required: true, trim: true },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true },
    email: { 
        type: String, 
        required: true, 
        trim: true, 
        lowercase: true, 
        match: [/^\S+@\S+\.\S+$/, "Invalid email format"] 
    },
    phone: { 
        type: String, 
        required: true, 
        match: [/^\d{10}$/, "Phone number must be 10 digits"]
    },
    date: { 
        type: String, 
        required: true, 
        match: [/^\d{4}-\d{2}-\d{2}$/, "Date must be in YYYY-MM-DD format"]
    },
    time: { type: String, required: true },
    type: { type: String, required: true, enum: ["General", "Emergency", "VIP"] },
    status: { type: String, enum: ["Pending", "Booked", "Cancelled"], default: "Pending" },
    department: { type: String, required: true },
    reached: { type: Boolean, default: false }, 
    doctorId: { type: mongoose.Schema.Types.ObjectId, ref: 'Doctor', required: true },
    doctorName: { type: String, required: true },
}, { timestamps: true }); // Auto-adds createdAt and updatedAt fields

const Appointment = mongoose.model('Appointment', appointmentSchema);

module.exports = Appointment;
