const mongoose = require("mongoose");

const bedSchema = new mongoose.Schema({
  bedNumber: { type: String, required: true, unique: true },
  number: { type: Number, required: true },
  type: { type: String, enum: ["ICU", "Private", "General"], required: true },
  status: { type: String, enum: ["Occupied", "Free"], default: "Free" },
});

module.exports = mongoose.model("Bed", bedSchema);
