// models/Inventory.js
const mongoose = require('mongoose');

const InventorySchema = new mongoose.Schema({
  name: { type: String, required: true },
  quantity: { type: Number, required: true },
  manufacturer: { type: String },
  expiry_date: { type: Date },
  manufacturing_date: { type: Date, required: true },
  cost: { type: Number, required: true },
});

const Inventory = mongoose.model('Inventory', InventorySchema);
module.exports = Inventory;