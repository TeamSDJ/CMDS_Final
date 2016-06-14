var myMap;
var myMode     = 'WALKING';
var myLatLng   = {lng: 121.53, lat: 25.03};
var myContent  = 'Map Marker';
var myDistance = 1000; // meters
var myToken    = 'EAALoPr8WJz4BACuIz1hSKsDdez4DRztUghLXsEzQKZB92ssnv8kOR1YYdUB2ZCTU1i3DLHRSkWZA3eH0ZBYOnZBPZCGKaBX7H17Kv6wtPQYUEhgjkrVb7LtOXo4m5OodgNRZCKVLXZAnZAAmGbvZCaonfz33ZAmW4wsiKQZD';

function runAllProcess() {
  myMap = createMap();
  setMarkerInfo(myMap);
  getPlaces(myMap);
  searchByPlace(myMap);
  FBGraphSearch();
}

/*************************
 Map Create
 *************************/
function createMap() {
  return new google.maps.Map(document.getElementById('map'), {
    zoom:   16,
    center: myLatLng
  });
}

/*************************
 Marker Information
 *************************/
function setMarkerInfo(map) {
  var marker = new google.maps.Marker({
    map:       map,
    title:     'Map Marker',
    position:  myLatLng,
    draggable: true,
    animation: google.maps.Animation.DROP,
  });
  var infowindow = new google.maps.InfoWindow({
    content: myContent
  });
  marker.addListener('click', function () {
    toggleBounce(marker);
    infowindow.open(map, marker);
  });
}

function toggleBounce(marker) {
  if (marker.getAnimation() !== null) {
    marker.setAnimation(null);
  } else {
    marker.setAnimation(google.maps.Animation.BOUNCE);
  }
}

function getDefaultMarkerInfo() {
  return (
    '<div id="content">' +
      '<div id="siteNotice"></div>' +
      '<h1 id="firstHeading" class="firstHeading">Uluru</h1>' +
      '<div id="bodyContent">' +
        '<p>' +
          '<b>Uluru</b>, also referred to as <b>Ayers Rock</b>, is a large ' +
          'sandstone rock formation in the southern part of the ' +
          'Northern Territory, central Australia. It lies 335&#160;km (208&#160;mi) ' +
          'south west of the nearest large town, Alice Springs; 450&#160;km ' +
          '(280&#160;mi) by road. Kata Tjuta and Uluru are the two major ' +
          'features of the Uluru - Kata Tjuta National Park. Uluru is ' +
          'sacred to the Pitjantjatjara and Yankunytjatjara, the ' +
          'Aboriginal people of the area. It has many springs, waterholes, ' +
          'rock caves and ancient paintings. Uluru is listed as a World ' +
          'Heritage Site.' +
        '</p>' +
        '<p>' +
          'Attribution: Uluru, <a href="https://en.wikipedia.org/w/index.php?title=Uluru&oldid=297882194">' +
          'https://en.wikipedia.org/w/index.php?title=Uluru</a> ' +
          '(last visited June 22, 2009).' +
        '</p>' +
      '</div>' +
    '</div>'
  )
}

/*************************
 Get Places
 *************************/
function getPlaces(map) {
  var request = {
    location: myLatLng,
    radius:   '500',
    types:    ['store']
  };
  var service = new google.maps.places.PlacesService(map);
  service.nearbySearch(request, processResults);
}

function processResults(results, status, pagination) {
  if (status !== google.maps.places.PlacesServiceStatus.OK) return;
  else {
    createMarkers(results);
    if (pagination.hasNextPage) {
      var moreButton = document.getElementById('more');
      if (!moreButton) return;
      moreButton.disabled = false;
      moreButton.addEventListener('click', function () {
        moreButton.disabled = true;
        pagination.nextPage();
      });
    }
  }
}

function createMarkers(places) {
  var bounds = new google.maps.LatLngBounds();
  var placesList = document.getElementById('places');

  for (var i = 0, place; place = places[i]; i++) {
    console.log('lat:', place.geometry.location.lat(), ', lng:', place.geometry.location.lng());
    var image = {
      url:  place.icon,
      size: new google.maps.Size(71, 71),
      origin: new google.maps.Point(0, 0),
      anchor: new google.maps.Point(17, 34),
      scaledSize: new google.maps.Size(25, 25)
    };

    var marker = new google.maps.Marker({
      map: myMap,
      icon: image,
      title: place.name,
      position: place.geometry.location
    });
    if (placesList) placesList.innerHTML += '<li>' + place.name + '</li>';

    bounds.extend(place.geometry.location);
  }
  myMap.fitBounds(bounds);
}

/*************************
 Search by Place
 *************************/
function searchByPlace(map) {
  // Create the search box and link it to the UI element.
  var input = document.getElementById('pac-input');
  var markers = [];
  var searchBox = new google.maps.places.SearchBox(input);
  map.controls[google.maps.ControlPosition.TOP_LEFT].push(input);
  // Bias the SearchBox results towards current map's viewport.
  map.addListener('bounds_changed', function () {
    searchBox.setBounds(map.getBounds());
  });
  // [START region_getplaces]
  // Listen for the event fired when the user selects a prediction and retrieve
  // more details for that place.
  searchBox.addListener('places_changed', function () {
    var places = searchBox.getPlaces();
    if (places.length == 0) return;
    // Clear out the old markers.
    markers.forEach(function (marker) {
      marker.setMap(null);
    });
    markers = [];
    // For each place, get the icon, name and location.
    var bounds = new google.maps.LatLngBounds();
    places.forEach(function (place) {
      var placeLatLng = {lng: place.geometry.location.lng(), lat: place.geometry.location.lat()};
      calcRoute(myLatLng, placeLatLng, google.maps.TravelMode[myMode], map);
      FBGraphSearch(placeLatLng['lat'], placeLatLng['lng'], myDistance);
      outputPlaceInfo(map, place);

      var icon = {
        url: place.icon,
        size: new google.maps.Size(71, 71),
        origin: new google.maps.Point(0, 0),
        anchor: new google.maps.Point(17, 34),
        scaledSize: new google.maps.Size(25, 25)
      };

      // Create a marker for each place.
      markers.push(new google.maps.Marker({
        map: map,
        //icon: icon,
        title: place.name,
        position: place.geometry.location
      }));

      if (place.geometry.viewport) {
        // Only geocodes have viewport.
        bounds.union(place.geometry.viewport);
      } else {
        bounds.extend(place.geometry.location);
      }
    });
    map.fitBounds(bounds);
    map.setZoom(16);
  });
  // [END region_getplaces]
}

function outputPlaceInfo(map, place) {
  // console.log('===========================');
  // console.log('name:', place.name);
  // console.log('lat:', place.geometry.location.lat(), ', lng:', place.geometry.location.lng());
  // console.log('types:', place.types);
  // console.log('rating:', place.rating);
  // if (place.reviews)
  //   for (var i = 0; i < place.reviews.length; ++i) {
  //     console.log('review ' + String(i + 1) + ':', place.reviews[i].text);
  //     console.log('review type ' + String(i + 1) + ':', place.reviews[i].aspects[0]['type']);
  //   }
  // console.log('url:', place.url);
  // console.log('website:', place.website);
  // console.log('place id:', place.place_id);
  // console.log('===========================');
  var infowindow = new google.maps.InfoWindow();
  var service = new google.maps.places.PlacesService(map);
  service.getDetails({ placeId: place.place_id }, callback);
  function callback(place, status) {
    if (status == google.maps.places.PlacesServiceStatus.OK) {
      var marker = new google.maps.Marker({
        map: map,
        position: place.geometry.location
      });
      google.maps.event.addListener(marker, 'click', function() {
        infowindow.setContent(place.name);
        infowindow.open(map, this);
      });
    }
  }
}

/*************************
 Route Travel
 *************************/
function calcRoute(start, end, mode, map) {
  var directionsDisplay = new google.maps.DirectionsRenderer();
  var directionsService = new google.maps.DirectionsService();
  directionsDisplay.setMap(map);
  var request = {
    origin: start,
    destination: end,
    travelMode: mode
  };
  directionsService.route(request, function(response, status) {
    if (status == google.maps.DirectionsStatus.OK) {
      directionsDisplay.setDirections(response);
      console.log(response.routes[0].legs[0]['duration']['text']);
    }
  });
}

/*************************
 Facebook Graph API Search
 *************************/
function FBGraphSearch() {
  FB.api('/search?type=place&center=' + String(myLatLng['lat']) + ',' + String(myLatLng['lng']) + '&distance=' + String(myDistance) + '&access_token=' + myToken, function(response) { console.log(response); });
}

/*************************
 Original Project Code
 *************************/
//Create a single global variable
// var MAPAPP = {};
// MAPAPP.markers = [];
// MAPAPP.currentInfoWindow;
// MAPAPP.pathName = window.location.pathname;
// $(document).ready(function() {
  // initialize();
  // populateMarkers(MAPAPP.pathName);
  // var map = createMap(myLatLng);
  // setMarkerInfo(map, myLatLng, contentString);
  // getPlaces(map, myLatLng);
  // searchByPlace(map);
// });

/*************************
 Initialize our Google Map
 *************************/
// function initialize() {
//   var center = new google.maps.LatLng(39.9543926, -75.1627432);
//   var mapOptions = {
//     zoom: 13,
//     mapTypeId: google.maps.MapTypeId.ROADMAP,
//     center: center,
//   };
//   this.map = new google.maps.Map(document.getElementById('map_canvas'), mapOptions);
// };

/*************************
 Fill map with markers
 *************************/
// function populateMarkers(dataType) {
//   apiLoc = typeof apiLoc !== 'undefined' ? apiLoc : '/data/' + dataType + '.json';
//   // jQuery AJAX call for JSON
//   $.getJSON(apiLoc, function(data) {
//     //For each item in our JSON, add a new map marker
//     $.each(data, function(i, ob) {
//       var marker = new google.maps.Marker({
//         map: map,
//         position: new google.maps.LatLng(this.location.coordinates[0], this.location.coordinates[1]),
//         shopname: this.shopname,
//         details: this.details,
//         website: this.website,
//         icon: 'http://maps.google.com/mapfiles/ms/icons/red-dot.png'
//       });
//       //Build the content for InfoWindow
//       var content = '<h1 class="mt0"><a href="' + marker.website + '" target="_blank" title="' + marker.shopname + '">' + marker.shopname + '</a></h1><p>' + marker.details + '</p>';
//       marker.infowindow = new google.maps.InfoWindow({
//         content: content,
//         maxWidth: 400
//       });
//       //Add InfoWindow
//       google.maps.event.addListener(marker, 'click', function() {
//         if (MAPAPP.currentInfoWindow) MAPAPP.currentInfoWindow.close();
//         marker.infowindow.open(map, marker);
//         MAPAPP.currentInfoWindow = marker.infowindow;
//       });
//       MAPAPP.markers.push(marker);
//     });
//   });
// };
