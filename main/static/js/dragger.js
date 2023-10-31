class DeckDragger{
    elname = ''
    element = null;
    constructor(elselect, func){
        let element = $(elselect);

        element.on('mousedown', function(ev) {
            let elleft = element.offset().left;
            let eltop = element.offset().top;

            let movestartx = ev.pageX;
            let movestarty = ev.pageY;
            
            let cx = movestartx - elleft
            let cy = movestarty - eltop
            
            element.append("<div id='idname' style='background-color:green; position:absolute; top:"+cy+"px; left:"+cx+"px'></div>")
            element.on('mousemove', function(e) {
                e.preventDefault();

                let dx = e.clientX - movestartx;
                let dy = e.clientY - movestarty;


                let cel = $(elselect + " #idname")
                cel.css({
                    'width':dx,
                    'height':dy
                })
                func(dx, dy);
            });
        });
        
        element.on('mouseup', function(e) {
            element.off('mousemove');
        });
    }
    name(params) {
        
    }

}

class MobDragger{
    elname = ''
    element = null;
    constructor(elselect, func){
        let element = $(elselect);

        element.on('touchstart', function(ev) {
            let elleft = element.offset().left;
            let eltop = element.offset().top;

            let movestartx = ev.touches[0].clientX
            let movestarty = ev.touches[0].clientY;

            let cx = movestartx - elleft
            let cy = movestarty - eltop
            
            element.append("<div id='idname' style='background-color:green; transform-origin: 0% 100%; position:absolute; top:"+cy+"px; left:"+cx+"px'></div>")
            
            element.on('touchmove', function(e) {
                e.preventDefault();

                let dx = e.touches[0].clientX - movestartx;
                let dy = e.touches[0].clientY - movestarty;


                let cel = $(elselect + " #idname")
                cel.css({
                    'width':dx,
                    'height':dy
                })
                func(dx, dy);
            });
        
        });
        
        element.on('touchend', function(e) {
            element.off('touchend');
        });
    }
    name(params) {
        
    }

}


// new MobDragger('.imagepack', function(vx, vy){
//     // console.log(vx);
// });
