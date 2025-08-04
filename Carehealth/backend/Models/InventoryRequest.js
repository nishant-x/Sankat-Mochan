const mongoose = require("mongoose");

const inventoryRequestSchema = new mongoose.Schema(
  {
    doctorName: { type: String, required: true, trim: true },
    department: { type: String, required: true, trim: true },
    requests: [
      {
        medicine_name: { type: String, required: true, trim: true },
        quantity: { type: Number, required: true, min: 1 },
      },
    ],
    request_date: { type: Date, default: Date.now },
    handled_by_pharmacist: { type: Boolean, default: false },
    grand_total: { type: Number, required: true, min: 0 },
  },
  { timestamps: true }
);

const InventoryRequest = mongoose.model("InventoryRequest", inventoryRequestSchema);

module.exports = InventoryRequest; // ✅ Export as default
