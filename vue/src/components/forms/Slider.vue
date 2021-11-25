<template>
  <div class="t-field">
    <div class="t-field__label">Train / Val</div>
    <div class="slider" @mouseleave="stopDrag" @mouseup="stopDrag" ref="slider">
      <div class="slider__inputs">
        <input name="[info][part][train]" type="number" :value="btnFirstVal" :data-degree="degree" />
        <input name="[info][part][validation]" type="number" :value="100 - btnFirstVal" :data-degree="degree" />
      </div>
      <div class="slider__scales">
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
      <div class="slider__between" ref="between">
        <button
          class="slider__btn-1"
          :style="sliderFirstStyle"
          @mousedown="startDragFirst"
          @mouseup="stopDragFirst"
        ></button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Slider',
  data: () => ({
    input: 0,
    select: 0,
    btnFirstVal: 70,
    firstBtnDrag: false,
    key1: 1,
    key2: 1,
  }),
  props: {
    degree: Number,
  },
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
        var btn = document.querySelector('.slider__btn-1');
        let pos = e.pageX - btn.parentNode.getBoundingClientRect().x;
        this.btnFirstVal = Math.round((pos / 231) * 100);
        if (this.btnFirstVal < 10) this.btnFirstVal = 10;
        if (this.btnFirstVal > 90) this.btnFirstVal = 90;
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
    border-radius: 0 4px 4px 0;
    width: 27%;
  }
  &__first,
  &__second {
    font-size: 12px;
    line-height: 24px;
    input {
      height: 100%;
      background-color: #17212b00;
      border: none;
      padding: 0;
    }
  }
}
</style>