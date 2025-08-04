const mongoose = require("mongoose");

const DischargeBillSchema = new mongoose.Schema({
  name: { type: String, required: true },
  contact: { type: String, required: true },
  department: { type: String, required: true },
  bedType: { type: String, required: true },
  bedNumber: { type: String, required: true },
  allocationTime: { type: Date, required: true },
  dischargeTime: { type: Date, default: Date.now },
  daysStayed: { type: Number, required: true },
  totalCost: { type: Number, required: true },
  paymentStatus: { type: String, default: "Not Paid" },
});

module.exports = mongoose.model("DischargeBill", DischargeBillSchema);
