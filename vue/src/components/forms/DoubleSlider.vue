<template>
    <div class="range-slider" @mousemove="slider">
      <div class="sliders">
        <div class="first-slider" @mousedown="startDrag($event, 'first')" :style="firstSlider"></div>
        <div class="second-slider" @mousedown="startDrag($event,'second')" :style="secondSlider"></div>
      </div>
      <div class="scale">
        <div id="first-scale" :style="firstScale">{{ sliders.first }}</div>
        <div id="second-scale" :style="secondScale">{{ sliders.second - sliders.first }}</div>
        <div id="third-scale" :style="thirdScale">{{ 100 - sliders.second }}</div>
      </div>
      <div class="inputs">
        <input type="number" class="first-value" v-model="sliders.first">
        <input type="number" class="second-value" v-model="sliders.second">
      </div>
    </div>
</template>

<script>
export default {
  name: "DoubleSlider",
  data: () => ({
    dragging: false,
    draggingObj: " ",
    sliders: {
      first: 50,
      second: 77
    },
  }),
  computed: {
    firstScale() {
      return {
        width: this.sliders.first + '%'
      };
    },
    secondScale(){
      return {
        width: this.sliders.second - this.sliders.first + '%'
      }
    },
    thirdScale(){
      return {
        width: 100 - this.sliders.second + '%'
      }
    },
    firstSlider(){
      return {
        'margin-left': this.sliders.first + '%'
      }
    },
    secondSlider(){
      return {
        'margin-left': this.sliders.second + '%'
      }
    }
  },
  methods: {
     startDrag(event, block) {
      this.dragging = true;
      this.draggingObj = block
      this.CurrentX = event.x;
    },
    stopDrag() {
      this.dragging = false;
      this.draggingObj = null;
    },
    slider(event){
       if(this.dragging){
         let slider = document.querySelector(`.${this.draggingObj}-slider`);
         let pos = event.x - slider.parentNode.getBoundingClientRect().x
         this.sliders[this.draggingObj] = Math.round(pos / 231 * 100);
       }
       if(this.sliders.first < 5) this.sliders.first = 5;
       if(this.sliders.second > 95) this.sliders.second = 95;
       if(this.sliders.first > this.sliders.second - 5) this.sliders.first = this.sliders.second - 5;
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
    div{
      text-align: center;
      line-height: 24px;
    }
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
    display: none;
  }
</style>