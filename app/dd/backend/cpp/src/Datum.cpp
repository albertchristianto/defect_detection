#include "datum.hpp"

namespace dd {
    //constructor
    Datum::Datum() :
        timeStamp{ 0ull },
        className{ "Unknown" },
        confScore{ 0.0f },
        nf::async::BaseDatum{}
    {}
    //destructor
    Datum::~Datum()
    {}
    //copy constructor
    Datum::Datum(const Datum& other) :
        timeStamp{ other.timeStamp },
        className{ other.className },
        confScore{ other.confScore },
        spawnTime{ other.spawnTime },
        cvInputData{ other.cvInputData },
        nf::async::BaseDatum{ other }
    {}
    //copy assignment
    Datum& Datum::operator=(const Datum& other) {
        timeStamp = other.timeStamp;
        className = other.className;
        confScore = other.confScore;
        spawnTime = other.spawnTime;
        cvInputData = other.cvInputData;
        Finished = other.Finished.load();
        ForceForward = other.ForceForward.load();

        return *this;
    }
    //move constructor
    Datum::Datum(Datum&& other) :
        timeStamp{ other.timeStamp },
        className{ other.className },
        confScore{ other.confScore },
        spawnTime{ other.spawnTime },
        cvInputData{ other.cvInputData },
        nf::async::BaseDatum{ std::move(other) }
    {}
    //move assignment
    Datum& Datum::operator=(Datum&& other) {
        timeStamp = other.timeStamp;
        className = other.className;
        confScore = other.confScore;
        spawnTime = other.spawnTime;
        cvInputData = other.cvInputData;
        Finished = other.Finished.load();
        ForceForward = other.ForceForward.load();

        return *this;
    }
    //clone functions
    Datum Datum::clone() const {
        Datum other;
        other.timeStamp = timeStamp;
        other.className = className;
        other.confScore = confScore;
        other.spawnTime = spawnTime;
        other.cvInputData = cvInputData;
        other.Finished = this->Finished.load();
        other.ForceForward = this->ForceForward.load();

        return other;
    }
}