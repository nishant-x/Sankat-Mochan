const express = require("express");
const router = express.Router();
const appointments = require("../Models/AppointmentSchema");


router.get('/totalappoinments', (req , res) =>{

    appointments.countDocuments({}, (err, count) => {
        if (err) return res.status(500).send(err);
        res.send({ totalAppointments: count });
    });

})

module.exports = router;