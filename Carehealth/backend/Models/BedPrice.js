const mongoose = require("mongoose");

const bedPriceSchema = new mongoose.Schema({
  bedType: {
    type: String,
    required: true,
    unique: true,
    enum: ["ICU", "Private", "General"], 
  },
  pricePerDay: { 
    type: Number, 
    required: true,
    min: 0 ,
    default: 0
  },
});

module.exports = mongoose.model("BedPrice", bedPriceSchema);
