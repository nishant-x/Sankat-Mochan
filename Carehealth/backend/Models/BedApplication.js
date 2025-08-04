const mongoose = require("mongoose");

const applicationSchema = new mongoose.Schema({
  name: { type: String, required: true },
  contact: { type: String, required: true },
  department: {
    type: String,
    enum: ["General Medicine", "Cardiology", "Neurology", "Orthopedics", "Pediatrics"],
    required: true,
  },
  bedType: { type: String, enum: ["ICU", "Private", "General"], required: true },
  bedNumber: { type: String, required: true },
  allocationTime: { type: Date, default: Date.now },
  doctorId: { type: mongoose.Schema.Types.ObjectId, required: true},
  doctorname: { type: String, required: true}
});

module.exports = mongoose.model("BedApplication", applicationSchema);
// This is the schema for the application model. It defines the structure of the data that will be stored in the database.