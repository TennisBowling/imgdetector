#include <iostream>
#include <vector>
#include <string>
#include <cpr/cpr.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "Simple-Web-Server/server_http.hpp"
#include <SQLiteCpp/SQLiteCpp.h>
using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;
using json = nlohmann::json;

struct Hist
{
    cv::Mat red_hist;
    cv::Mat green_hist;
    cv::Mat blue_hist;

    bool operator==(const Hist &other) const
    {
        return (cv::countNonZero(red_hist != other.red_hist) == 0) && (cv::countNonZero(green_hist != other.green_hist) == 0) && (cv::countNonZero(blue_hist != other.blue_hist) == 0);
    }
};

Hist make_hist(cv::Mat &image)
{
    // make three histograms for red, green and blue
    cv::Mat red_hist, green_hist, blue_hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    int channels[] = {0};
    cv::calcHist(&image, 1, channels, cv::Mat(), red_hist, 1, &histSize, &histRange);
    channels[0] = 1;
    cv::calcHist(&image, 1, channels, cv::Mat(), green_hist, 1, &histSize, &histRange);
    channels[0] = 2;
    cv::calcHist(&image, 1, channels, cv::Mat(), blue_hist, 1, &histSize, &histRange);
    Hist hist;
    hist.red_hist = red_hist;
    hist.green_hist = green_hist;
    hist.blue_hist = blue_hist;
    return hist;
}

int main(int argc, char *argv[])
{

    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] - %v"); // nice style that i like

    HttpServer server;
    server.config.port = 8000;
    server.config.address = "0.0.0.0";
    std::vector<Hist> known_images; // the known images are the ones we will check against in the /check endpoint

    SQLite::Database db("./db.sqlite", SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
    db.exec("CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, rawimg TEXT);");

    // get all the images from the database
    SQLite::Statement query(db, "SELECT * FROM images;");

    while (query.executeStep())
    {
        // get the image and load into the Hist struct and add to vector
        std::string rawimg = query.getColumn("rawimg").getString();
        std::vector<unsigned char> image_data(rawimg.begin(), rawimg.end());
        cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

        Hist hist = make_hist(hsv);
        known_images.push_back(hist);
    }

    spdlog::info("Loaded {} known images", known_images.size());

    server.resource["/set_recognized"]["POST"] = [&known_images, &db](std::shared_ptr<HttpServer::Response> response, std::shared_ptr<HttpServer::Request> request)
    {
        json j = json::parse(request->content.string());

        spdlog::debug("Got request to set imgage {} as recognized", j["url"]);

        cpr::Response r = cpr::Get(cpr::Url{j["url"].get<std::string>()});
        std::vector<unsigned char> image_data(r.text.begin(), r.text.end());

        // load raw image into opencv, convert to hsv, calculate histogram, and store the histogram in the known_images vector
        // do this for red, green, and blue
        cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);

        cv::Mat hsv;
        try
        {
            cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        }
        catch (cv::Exception &e)
        {
            spdlog::error("Error converting image to hsv: {}", e.what());
            response->write(SimpleWeb::StatusCode::client_error_bad_request, "Error converting image to hsv");
            return;
        }

        Hist hist = make_hist(hsv);

        if (std::find(known_images.begin(), known_images.end(), hist) == known_images.end())
        {
            // if the histogram is not in the known_images vector, add it to the database
            SQLite::Statement insert(db, "INSERT INTO images (rawimg) VALUES (?);");
            insert.bind(1, r.text);
            insert.exec();
            known_images.push_back(hist);
            response->write(SimpleWeb::StatusCode::success_ok, json({{"success", true}, {"error", "none"}}).dump());
        }
        else
        {
            response->write(SimpleWeb::StatusCode::success_ok, json({{"success", false}, {"error", "image_already_recognized"}}).dump());
        }
    };

    server.resource["/check"]["GET"] = [&known_images](std::shared_ptr<HttpServer::Response> response, std::shared_ptr<HttpServer::Request> request)
    {
        json j = json::parse(request->content.string());
        spdlog::debug("Got check request for image url {}", j["url"]);

        cpr::Response r = cpr::Get(cpr::Url{j["url"].get<std::string>()});
        std::vector<unsigned char> image_data(r.text.begin(), r.text.end());

        cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
        cv::Mat hsv;

        try
        {
            cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        }
        catch (cv::Exception &e)
        {
            spdlog::error("Error converting image to hsv: {}", e.what());
            response->write(SimpleWeb::StatusCode::client_error_bad_request, "Error converting image to hsv");
            return;
        }

        Hist h = make_hist(hsv);
        // now compare the histogram of the image to the known_images vector

        for (auto &i : known_images)
        {
            double red_dist = cv::compareHist(h.red_hist, i.red_hist, cv::HISTCMP_BHATTACHARYYA);
            double green_dist = cv::compareHist(h.green_hist, i.green_hist, cv::HISTCMP_BHATTACHARYYA);
            double blue_dist = cv::compareHist(h.blue_hist, i.blue_hist, cv::HISTCMP_BHATTACHARYYA);

            if (red_dist < 0.25 && green_dist < 0.25 && blue_dist < 0.25)
            {
                response->write(SimpleWeb::StatusCode::success_ok, json({{"result", "match"}, {"distance", red_dist + green_dist + blue_dist}}).dump());
            }
        }
        response->write(SimpleWeb::StatusCode::success_ok, json({{"result", "no match"}}).dump());
    };

    server.resource["/get_recognized"]["GET"] = [&known_images, &db](std::shared_ptr<HttpServer::Response> response, std::shared_ptr<HttpServer::Request> request)
    {
        // send the known_images vector to the client
        spdlog::debug("sending known images to client, size: {}", known_images.size());

        std::vector<std::string> imgbuffer;

        SQLite::Statement query(db, "SELECT * FROM images;");

        while (query.executeStep())
        {
            // get the image and push back to vector buffer
            vector.push_back(query.getColumn("rawimg").getString());
        }

        json j;
        j["images"] = imgbuffer;
        response->write(SimpleWeb::StatusCode::success_ok, j.dump());
    };

    spdlog::info("Listening on {}:{}", server.config.address, server.config.port);
    server.start();
}