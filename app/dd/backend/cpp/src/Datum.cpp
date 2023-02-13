#include "datum.hpp"

namespace dd {
    //constructor
    Datum::Datum() :
        timeStamp{ 0ull },
        className{ "Unknown" },
        confScore{ 0.0f },
        nf::BaseDatum{}
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
        nf::BaseDatum{ other }
    {}
    //copy assignment
    Datum& Datum::operator=(const Datum& other):
        nf::BaseDatum{ other }
    {
        timeStamp = other.timeStamp;
        className = other.className;
        confScore = other.confScore;
        spawnTime = other.spawnTime;
        cvInputData = other.cvInputData;

        return *this;
    }
    //move constructor
    Datum::Datum(Datum&& other) :
        timeStamp{ other.timeStamp },
        className{ other.className },
        confScore{ other.confScore },
        spawnTime{ other.spawnTime },
        cvInputData{ other.cvInputData },
        nf::BaseDatum{ std::move(other) }
    {}
    //move assignment
    Datum& Datum::operator=(Datum&& other):
        nf::BaseDatum{ std::move(other) }
    {
        timeStamp = other.timeStamp;
        className = other.className;
        confScore = other.confScore;
        spawnTime = other.spawnTime;
        cvInputData = other.cvInputData;

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