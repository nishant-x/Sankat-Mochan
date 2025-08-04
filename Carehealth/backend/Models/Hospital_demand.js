const mongoose = require('mongoose');

const HospitalDemandSchema = new mongoose.Schema({
    request_id: { type: String, unique: true, required: true }, // Unique request identifier
    hospital_name: { type: String, required: true },
    items: [
        {
            name: { type: String, required: true },
            quantity: { type: Number, required: true },
        }
    ],
    status: { type: boolean, default: false },
    demand_date: { type: Date, default: Date.now } // Tracks when the request was made
});

module.exports = mongoose.model('HospitalDemand', HospitalDemandSchema);
