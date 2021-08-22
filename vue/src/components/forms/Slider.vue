<template>
<div class="t-field">
  <div class="t-field__label">Train / Val / Test</div>
  <div class="slider">
    <div class="slider__scales">
      <div class="scales__first"></div>
      <div class="scales__second"></div>
      <div class="scales__third"></div>
    </div>
    <div class="slider__between" ref="between">
      <button class="slider__btn-1" :style='sliderFirstStyle' @mousedown="firstBtn"></button>
      <button class="slider__btn-2" :style='sliderSecondStyle' @mousedown="secondBtn"></button>
    </div>
  </div>
</div>
</template>

<script>
export default {
  name: "Slider",
  data: () => ({
    btnFirstVal: 50,
    btnSecondVal: 77,
  }),
  props: {
    // degree: Number
  },
  methods: {
    firstBtn(e){
      var btn = e.target;

      document.onmousemove = function (e) {
        let pos = e.x - btn.parentNode.getBoundingClientRect().x;
        this.btnFirstVal = Math.round((pos / 231) * 100);
        console.log(this.btnFirstVal);
      }
      document.onmouseup = function() {
        document.onmousemove = document.onmouseup = null;
      };
      return false;
    },
    secondBtn(){},
  },
  computed: {
    sliderFirstStyle() {
      return {
        left: this.btnFirstVal + "%",
      };
    },
    sliderSecondStyle() {
      return {
        left: this.btnSecondVal + "%",
      };
    },
  },
}
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
.slider{
  margin-top: 10px;
  width: 231px;
  height: 24px;
  background: #0e1621;
  border-radius: 4px;
  &__between{
    position: absolute;
    display: flex;
    width: 231px;
  }
  &__btn-1, &__btn-2{
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
  &__scales{
    display: flex;
    width: 231px;
    height: 24px;
    position: absolute;
  }
}
.scales{
  &__first{
    background: #d6542c;
    border-radius: 4px 0 0 4px;
    width: 50%;
  }
  &__second{
    background: #609e42;
    width: 27%;
  }
  &__third{
    background: #5191f2;
    border-radius: 0 4px 4px 0;
    width: 23%;
  }
}
</style>