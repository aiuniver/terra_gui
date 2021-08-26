<template>
  <div class="t-field">
    <div class="t-field__label">Train / Val / Test</div>
    <div class="slider"
    @mouseleave="stopDrag"
    @mouseup="stopDrag"
    ref="slider"
    >
      <div class="slider__inputs">
        <input name="[info][part][train]" type="number" :value="btnFirstVal" :data-degree="degree" />
        <input
          name="[info][part][validation]"
          type="number"
          :value="btnSecondVal - btnFirstVal"
          :data-degree="degree"
        />
        <input name="[info][part][test]" type="number" :value="100 - btnSecondVal" :data-degree="degree" />
      </div>
      <div class="slider__scales" 
      >
        <div class="scales__first" :style="firstScale">{{ btnFirstVal }}</div>
        <div class="scales__second" :style="secondScale">{{ btnSecondVal - btnFirstVal }}</div>
        <div class="scales__third" :style="thirdScale">{{ 100 - btnSecondVal }}</div>
      </div>
      <div class="slider__between" ref="between">
        <button
          class="slider__btn-1"
          :style="sliderFirstStyle"
          @mousedown="startDragFirst"
          @mouseup="stopDragFirst"
        ></button>
        <button
          class="slider__btn-2"
          :style="sliderSecondStyle"
          @mousedown="startDragSecond"
          @mouseup="stopDragSecond"
        ></button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Slider',
  data: () => ({
    btnFirstVal: 50,
    btnSecondVal: 77,
    firstBtnDrag: false,
    secondBtnDrag: false,
  }),
  props: {
    degree: Number,
  },
  methods: {
    stopDrag() {
      this.$refs.slider.removeEventListener('mousemove', this.firstBtn);
      this.$refs.slider.removeEventListener('mousemove', this.secondBtn);
    },
    startDragFirst() {
      this.firstBtnDrag = true;
      this.$refs.slider.addEventListener('mousemove', this.firstBtn);
    },
    stopDragFirst() {
      this.$refs.slider.removeEventListener('mousemove', this.firstBtn);
      this.firstBtnDrag = false;
    },
    startDragSecond() {
      this.secondBtnDrag = true;
      this.$refs.slider.addEventListener('mousemove', this.secondBtn);
    },
    stopDragSecond() {
      this.$refs.slider.removeEventListener('mousemove', this.secondBtn);
      this.secondBtnDrag = false;
    },
    firstBtn(e) {
      if (this.firstBtnDrag) {
        var btn = document.querySelector('.slider__btn-1');
        let pos = e.pageX - btn.parentNode.getBoundingClientRect().x;
        this.btnFirstVal = Math.round((pos / 231) * 100);
        if (this.btnFirstVal < 5) this.btnFirstVal = 5;
        if (this.btnFirstVal > 95) this.btnFirstVal = 95;
        if (this.btnFirstVal > this.btnSecondVal - 5) this.btnFirstVal = this.btnSecondVal - 5;
      }
    },
    secondBtn(e) {
      if (this.secondBtnDrag) {
        var btn = document.querySelector('.slider__btn-2');
        let pos = e.pageX - btn.parentNode.getBoundingClientRect().x;
        this.btnSecondVal = Math.round((pos / 231) * 100);
        if (this.btnSecondVal < 5) this.btnSecondVal = 5;
        if (this.btnSecondVal > 95) this.btnSecondVal = 95;
        if (this.btnSecondVal < this.btnFirstVal + 5) this.btnSecondVal = this.btnFirstVal + 5;
      }
    },
  },
  computed: {
    sliderFirstStyle() {
      return {
        left: this.btnFirstVal + '%',
      };
    },
    sliderSecondStyle() {
      return {
        left: this.btnSecondVal + '%',
      };
    },
    firstScale() {
      return {
        width: this.btnFirstVal + '%',
      };
    },
    secondScale() {
      return {
        width: this.btnSecondVal - this.btnFirstVal + '%',
      };
    },
    thirdScale() {
      return {
        width: 100 - this.btnSecondVal + '%',
      };
    },
  },
};
</script>

<style scoped lang="scss">
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
.slider {
  margin-top: 10px;
  width: 231px;
  height: 24px;
  background: #0e1621;
  border-radius: 4px;
  &__between {
    position: absolute;
    display: flex;
    width: 231px;
  }
  &__btn-1,
  &__btn-2 {
    height: 24px;
    width: 2px;
    position: absolute;
    background: #ffffff;
    border: none;
    cursor: move;
    &:before {
      content: ' ';
      display: block;
      width: 6px;
      height: 6px;
      border: 1px solid #ffffff;
      border-radius: 4px;
      background: #17212b;
      position: relative;
      left: -2px;
    }
  }
  &__scales {
    display: flex;
    width: 231px;
    height: 24px;
    position: absolute;
    div {
      text-align: center;
    }
  }
  &__inputs {
    display: none;
  }
}
.scales {
  &__first {
    background: #d6542c;
    border-radius: 4px 0 0 4px;
    width: 50%;
  }
  &__second {
    background: #609e42;
    width: 27%;
  }
  &__third {
    background: #5191f2;
    border-radius: 0 4px 4px 0;
    width: 23%;
  }
  &__first,
  &__second,
  &__third {
    font-size: 12px;
    line-height: 24px;
  }
}
</style>