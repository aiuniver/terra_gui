<template>
    <div class="range-slider" >
      <div class="sliders">
        <div class="first-slider" @mousedown="startDrag" @mousemove="slider"><div></div></div>
        <div class="second-slider" @mousedown="startDrag"><div></div></div>
      </div>
      <div class="scale">
        <div id="first-scale"></div>
        <div id="second-scale"></div>
        <div id="third-scale"></div>
      </div>
    </div>
</template>

<script>
export default {
  name: "DoubleSlider",
  data: () => ({
    dragging: false,
    CurrentX: 0,
    minValue: 0,
    maxValue: 100,
    firstSlider: 50,
    secondSlider: 77,
  }),
  methods: {
     startDrag(event) {
      this.dragging = true;
      this.CurrentX = event.x;
    },
    stopDrag() {
      this.dragging = false;
    },
    slider(event){
       if(this.dragging){
         if(event.x > this.CurrentX){
           console.log(event.target);
           event.target.style.marginLeft = (parseInt((event.target.style.marginLeft) || parseInt(window.getComputedStyle(event.target).marginLeft))) + 2 + 'px';
         } else{
           event.target.style.marginLeft = (parseInt((event.target.style.marginLeft) || parseInt(window.getComputedStyle(event.target).marginLeft))) - 2 + 'px';
         }
       }
    }
  },
  mounted() {
    window.addEventListener('mouseup', this.stopDrag);
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
      div{
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
    margin-left: 27%;
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
</style>