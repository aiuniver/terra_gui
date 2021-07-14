<template>
  <div class="range-slider">
    <input type="range" min="0" max="100" step="1" v-model="sliderMin" />
    <input type="range" min="0" max="100" step="1" v-model="sliderMax" />
  </div>
</template>

<script>

export default {
  
  data: () => ({
    minAngle: 10,
    maxAngle: 30,
  }),
  computed: {
    sliderMin: {
      get() {
        var val = parseInt(this.minAngle);
        return val;
      },
      set(val) {
        // console.log('min', val)
        this.$emit('set', { min: this.minAngle, max: this.maxAngle })
        val = parseInt(val);
        if (val > this.maxAngle) {
          this.maxAngle = val;
        }
        this.minAngle = val;
      },
    },
    sliderMax: {
      get() {
        var val = parseInt(this.maxAngle);
        return val;
      },
      set(val) {
        // console.log('max', val)
        this.$emit('set', { min:  this.minAngle, max: this.maxAngle })
        val = parseInt(val);
        if (val < this.minAngle) {
          this.minAngle = val;
        }
        this.maxAngle = val;
      },
    },
  },
};
</script>

<style>
.range-slider {
  width: 100%;
  margin: auto;
  text-align: center;
  position: relative;
  height: 1em;
}

.range-slider input[type=range] {
  position: absolute;
  left: 0;
  bottom: 0;
}

input[type=range] {
  -webkit-appearance: none;
  width: 100%;
}

input[type=range]:focus {
  outline: none;
}

input[type=range]:focus::-webkit-slider-runnable-track {
  background: #6c7883;
}

input[type=range]:focus::-ms-fill-lower {
  background: #6c7883;
}

input[type=range]:focus::-ms-fill-upper {
  background: #6c7883;
}

input[type=range]::-webkit-slider-runnable-track {
  width: 100%;
  height: 3px;
  cursor: pointer;
  background: #6c7883;
  border-radius: 1px;
  box-shadow: none;
  border: 0;
}

input[type=range]::-webkit-slider-thumb {
  z-index: 2;
  position: relative;
  box-shadow: 0px 0px 0px #000;
  border: 1px solid #65b9f4;
  height: 12px;
  width: 12px;
  border-radius: 15px;
  background: #65b9f4;
  cursor: pointer;
  -webkit-appearance: none;
  margin-top: -4px;
}

</style>
