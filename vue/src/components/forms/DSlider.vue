<template>
  <div class="t-field">
    <div class="t-field__label">Train / Val</div>
    <div :class="['d-slider', { 'd-slider--disable': disable }]" @mouseleave="stopDrag" @mouseup="stopDrag" ref="slider">
      <div class="d-slider__inputs">
        <input name="[info][part][train]" type="number" :value="btnFirstVal" :data-degree="degree" />
        <input name="[info][part][validation]" type="number" :value="100 - btnFirstVal" :data-degree="degree" />
      </div>
      <div class="d-slider__scales">
        <div class="scales__first" :style="firstScale">
          <input
            :value="btnFirstVal"
            v-autowidth
            type="number"
            autocomplete="off"
            :key="key1"
            ref="key1"
            @keypress.enter="inter(1, $event)"
            @blur="clickInput(1, $event)"
            @focus="focus"
          />
        </div>
        <div class="scales__second" :style="secondScale">
          <input
            :value="100 - btnFirstVal"
            v-autowidth
            type="number"
            autocomplete="off"
            :key="key2"
            ref="key2"
            @keypress.enter="inter(2, $event)"
            @blur="clickInput(2, $event)"
            @focus="focus"
          />
        </div>
      </div>
      <div class="d-slider__between" ref="between">
        <button class="d-slider__btn-1" :style="sliderFirstStyle" @mousedown="startDragFirst" @mouseup="stopDragFirst"></button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'd-slider',
  props: {
    degree: Number,
    disable: Boolean,
    value: {
      type: Number,
      default: 70,
    },
  },
  data: () => ({
    input: 0,
    select: 0,
    // btnFirstVal: 70,
    firstBtnDrag: false,
    key1: 1,
    key2: 1,
  }),

  methods: {
    focus({ target }) {
      target.select();
    },
    inter(i, { target }) {
      target.blur();
      const ref = this.$refs[`key${i + 1}`];
      if (ref) {
        ref.focus();
        this.$nextTick(() => {
          ref.select();
        });
      }
    },
    clickInput(i, { target }) {
      const value = +target.value;
      if (i === 1) {
        if (value >= 0 && value <= 90) {
          this.btnFirstVal = value > 10 ? value : 10;
        }
      }
      if (i === 2) {
        if (value >= 0 && value <= 90) {
          this.btnFirstVal = value > 10 ? 100 - value : 10;
        }
      }
      this[`key${i}`] += 1;
    },
    stopDrag() {
      this.$refs.slider.removeEventListener('mousemove', this.firstBtn);
    },
    startDragFirst() {
      this.firstBtnDrag = true;
      this.$refs.slider.addEventListener('mousemove', this.firstBtn);
    },
    stopDragFirst() {
      this.$refs.slider.removeEventListener('mousemove', this.firstBtn);
      this.firstBtnDrag = false;
    },
    // startDragSecond() {
    //   this.secondBtnDrag = true;
    //   this.$refs.slider.addEventListener('mousemove', this.secondBtn);
    // },
    // stopDragSecond() {
    //   this.$refs.slider.removeEventListener('mousemove', this.secondBtn);
    //   this.secondBtnDrag = false;
    // },
    firstBtn(e) {
      if (this.firstBtnDrag) {
        var btn = document.querySelector('.d-slider__btn-1');
        let pos = e.pageX - btn.parentNode.getBoundingClientRect().x;
        const width = this.$refs.slider.clientWidth
        let value = Math.round((pos / width) * 100);
        if (value < 10) value = 10;
        if (value > 90) value = 90;
        // console.log(value)
        this.btnFirstVal = value
      }
    },
    diff(value, max = 90, min = 10) {
      if (value < min) {
        value = min;
      }
      if (value > max) {
        value = max;
      }
      return value;
    },
  },
  computed: {
    sliderFirstStyle() {
      return {
        left: this.diff(this.btnFirstVal, 90) + '%',
      };
    },
    sliderSecondStyle() {
      return {
        left: this.diff(this.btnFirstVal, 90) + '%',
      };
    },
    firstScale() {
      return {
        width: this.diff(this.btnFirstVal, 90) + '%',
      };
    },
    secondScale() {
      return {
        width: this.diff(100 - this.btnFirstVal, 90) + '%',
      };
    },
    btnFirstVal: {
      set(value) {
        this.$emit('input', (value / 100));
      },
      get() {
        return Math.round(this.value * 100);
      },
    },
  },
  watch: {
    disable(value) {
      if (value) {
        this.btnFirstVal = 0;
      } else {
        this.btnFirstVal = 70;
      }
    },
  },
};
</script>

<style scoped lang="scss">
.t-field {
  user-select: none;
  margin-bottom: 20px;
  position: relative;
  &__label {
    width: 150px;
    max-width: 330px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }
}
.d-slider {
  margin-top: 10px;
  width: 100%;
  height: 40px;
  background: #242f3d50;
  border-radius: 4px;
  position: relative;
  &__between {
    position: absolute;
    display: flex;
    // width: 231px;
    width: 100%;
  }
  &__btn-1,
  &__btn-2 {
    height: 40px;
    width: 2px;
    position: absolute;
    background: #ffffff;
    border: none;
    cursor: move;
    padding: 0;
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
    // width: 231px;
    width: 100%;
    height: 40px;
    position: absolute;
    div {
      text-align: center;
    }
  }
  &__inputs {
    display: none;
  }
  &--disable {
    cursor: default;
    opacity: 0.5;
    &::after {
      content: '';
      display: block;
      position: absolute;
      height: 100%;
      width: 100%;
    }
  }
}
.scales {
  &__first {
    border-radius: 4px 0 0 4px;
    border: 1px solid #d6542c;

    width: 50%;
  }
  &__second {
    border: 1px solid #609e42;
    border-radius: 0 4px 4px 0;
    width: 27%;
  }
  &__first,
  &__second {
    font-size: 12px;
    line-height: 24px;
    input {
      font-size: 1rem;
      height: 100%;
      background-color: #17212b00;
      border: none;
      padding: 0;
    }
  }
}
</style>