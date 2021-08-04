<template>
    <div class="range-slider" @mousemove="slider">
      <div class="sliders">
        <div class="first-slider" @mousedown="startDrag($event, 'first')"></div>
        <div class="second-slider" @mousedown="startDrag($event,'second')"></div>
      </div>
      <div class="scale">
        <div id="first-scale"></div>
        <div id="second-scale"></div>
        <div id="third-scale"></div>
      </div>
      <div class="inputs">
        <input type="number" class="first-value">
        <input type="number" class="second-value">
      </div>
    </div>
</template>

<script>
export default {
  name: "DoubleSlider",
  data: () => ({
    dragging: false,
    draggingObj: " ",
    CurrentX: 0,
    minValue: 0,
    maxValue: 100,
    firstSlider: 50,
    secondSlider: 77,
  }),
  methods: {
     startDrag(event, block) {
      this.dragging = true;
      this.draggingObj = block
      this.CurrentX = event.x;
    },
    stopDrag() {
      this.dragging = false;
      this.draggingObj = " ";
    },
    slider(event){
       let slider = document.querySelector(`.${this.draggingObj}-slider`),
             scale  = document.querySelector(`#${this.draggingObj}-scale`);
       if(this.dragging && slider.parentNode.getBoundingClientRect().x+5 < event.x && slider.parentNode.getBoundingClientRect().x + 225 > event.x){
         let pos = event.x - slider.parentNode.getBoundingClientRect().x
         slider.style.marginLeft = pos + 'px';
         if(this.draggingObj == 'first'){
           scale.style.width  = (pos / 231 * 100) + "%";
           document.querySelector('#second-scale').style.width = 100 - (pos / 231 * 100) - (document.querySelector('#third-scale').offsetWidth / 231 * 100) + "%";

           document.querySelector('.first-value').value = pos / 231 * 100;
           document.querySelector('.second-value').value = (pos / 231 * 100) + (100 - (pos / 231 * 100) - (document.querySelector('#third-scale').offsetWidth / 231 * 100));
         } else{
           scale.style.width = (pos / 231 * 100) - document.querySelector('#first-scale').offsetWidth / 231 * 100 + "%";
           document.querySelector('#third-scale').style.width = 100 - (document.querySelector('#second-scale').offsetWidth / 231 * 100) - (document.querySelector('#first-scale').offsetWidth / 231 * 100) + "%";

           document.querySelector('.first-value').value = document.querySelector('#first-scale').offsetWidth / 231 * 100;
           document.querySelector('.second-value').value = (pos / 231 * 100);
         }

       }
    }
  },
  mounted() {
    window.addEventListener('mouseup', this.stopDrag);
  },
  updated() {

  }
}
</script>

<style lang="scss" scoped>
  .range-slider{
    padding-top: 10px;
    input{
      background: none;
      position: absolute;
    }
  }
  .sliders{
    display: flex;
    position: absolute;
    width: 231px;
    div{
      height: 24px;
      width: 2px;
      background: #FFFFFF;
      cursor: pointer;
      position: absolute;
      &:before{
        content: ' ';
        display: block;
        width: 6px;
        height: 6px;
        border: 1px solid #FFFFFF;
        border-radius: 4px;
        background: #17212B;
        position: relative;
        top: 9px;
        left: -2px;
      }
    }
  }
  .first-slider{
    margin-left: 50%;
  }
  .second-slider{
    margin-left: 77%;
  }
  .scale{
    height: 24px;
    width: 231px;
    display: flex;
    #first-scale{
      background: #D6542C;
      border-radius: 4px 0 0 4px;
      width: 50%;
    }
    #second-scale{
      background: #609E42;
      width: 27%;
    }
    #third-scale{
      background: #5191F2;
      border-radius: 0 4px 4px 0;
      width: 23%;
    }
  }
  .inputs{
    display: flex;
    input{
      //display: none;
      width: 60px;
      height: 24px;
      padding: 4px;
      &:nth-child(2){
        margin-left: 70px;
      }
    }
  }
</style>