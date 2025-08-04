const mongoose = require('mongoose');

const patientPrescriptionSchema = new mongoose.Schema({
  patient_name: { type: String, required: true, trim: true },
  doctor_name: { type: String, required: true, trim: true },
  prescription: [
    {
      medicine_name: { type: String, required: true, trim: true },
      quantity: { type: Number, required: true, min: 1 },
      dosage: { type: String, required: true, trim: true },
      duration: { type: String, trim: true }, // Optional: Duration of the medication
    },
  ],
  prescription_date: { type: Date, required: true, default: Date.now }, // Renamed for clarity
  doctor_id: { type: mongoose.Schema.Types.ObjectId, required: true, ref: 'Doctor' },
  patient_id: { type: mongoose.Schema.Types.ObjectId, required: true, ref: 'Patient' }, // Ensure the `Patient` collection exists
  handled_by_pharmacist: { type: Boolean, default: false },
  grand_total: { type: Number, default: 0, min: 0 }, // Ensure this is updated accurately
});

const PatientPrescription = mongoose.model('PatientPrescription', patientPrescriptionSchema);

module.exports = PatientPrescription;
