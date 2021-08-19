<template>
  <div class="t-field">
    <label class="t-field__label">Train / Val / Test</label>
    <div class="slider">
      <div class="range-slider" @mousemove="slider">
        <div class="sliders">
          <div class="first-slider" @mousedown="startDrag($event, 'first')" :style="firstSlider"></div>
          <div class="second-slider" @mousedown="startDrag($event, 'second')" :style="secondSlider"></div>
        </div>
        <div class="scale">
          <div id="first-scale" :style="firstScale">{{ sliders.first }}</div>
          <div id="second-scale" :style="secondScale">{{ sliders.second - sliders.first }}</div>
          <div id="third-scale" :style="thirdScale">{{ 100 - sliders.second }}</div>
        </div>
        <div class="inputs">
          <input name="[info][part][train]" type="number" :value="sliders.first" :data-degree="degree" />
          <input name="[info][part][validation]" type="number" :value="sliders.second - sliders.first" :data-degree="degree" />
          <input name="[info][part][test]" type="number" :value="100 - sliders.second" :data-degree="degree" />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DoubleSlider',
  props: {
    degree: Number
  },
  data: () => ({
    dragging: false,
    draggingObj: null,
    sliders: {
      first: 50,
      second: 77,
    },
  }),
  computed: {
    firstScale() {
      return {
        width: this.sliders.first + '%',
      };
    },
    secondScale() {
      return {
        width: this.sliders.second - this.sliders.first + '%',
      };
    },
    thirdScale() {
      return {
        width: 100 - this.sliders.second + '%',
      };
    },
    firstSlider() {
      return {
        'margin-left': this.sliders.first + '%',
      };
    },
    secondSlider() {
      return {
        'margin-left': this.sliders.second + '%',
      };
    },
  },
  methods: {
    startDrag(event, block) {
      this.dragging = true;
      this.draggingObj = block;
      this.CurrentX = event.x;
    },
    stopDrag() {
      this.dragging = false;
      this.draggingObj = null;
    },
    slider(event) {
      event.preventDefault();
      if (this.dragging) {
        if (this.sliders.first < 10){
          this.sliders.first = 10;
          return;
        }
        else if (this.sliders.second > 90){
          this.sliders.second = 90;
          return;
        }
        else if (this.sliders.first > this.sliders.second - 10){
          if(this.draggingObj == "first") --this.sliders.first;
          else ++this.sliders.second;
          return;
        }
        let slider = document.querySelector(`.${this.draggingObj}-slider`);
        let pos = event.x - slider.parentNode.getBoundingClientRect().x;
        this.sliders[this.draggingObj] = Math.round((pos / 231) * 100);
      }
    },
  },
  mounted() {
    window.addEventListener('mouseup', this.stopDrag);
  },
  destroyed() {
    window.removeEventListener('mouseup', this.stopDrag);
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  margin-bottom: 20px;
  &__label {
    width: 150px;
    max-width: 330px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.5;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }
}
.range-slider {
  padding-top: 10px;
  input {
    background: none;
    position: absolute;
  }
}
.sliders {
  display: flex;
  position: absolute;
  width: 231px;
  div {
    height: 24px;
    width: 2px;
    background: #ffffff;
    cursor: pointer;
    position: absolute;
    &:before {
      content: ' ';
      display: block;
      width: 6px;
      height: 6px;
      border: 1px solid #ffffff;
      border-radius: 4px;
      background: #17212b;
      position: relative;
      top: 9px;
      left: -2px;
    }
  }
}
.first-slider {
  margin-left: 50%;
}
.second-slider {
  margin-left: 77%;
}
.scale {
  height: 24px;
  width: 231px;
  display: flex;
  div {
    text-align: center;
    line-height: 24px;
    font-size: 12px;
  }
  #first-scale {
    background: #d6542c;
    border-radius: 4px 0 0 4px;
    width: 50%;
  }
  #second-scale {
    background: #609e42;
    width: 27%;
  }
  #third-scale {
    background: #5191f2;
    border-radius: 0 4px 4px 0;
    width: 23%;
  }
}
.inputs {
  display: none;
}
</style>