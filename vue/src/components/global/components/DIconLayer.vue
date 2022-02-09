<template>
  <svg width="30" height="30" viewBox="0 0 30 30" fill="none" xmlns="http://www.w3.org/2000/svg">
    <g clip-path="url(#clip0_3213:3118)">
      <path
        d="M5.38697 9.86065L14.9993 2.38435L24.6094 9.85884L23.3994 10.7958L23.3979 10.797L14.999 17.3358L6.58901 10.7972L6.5884 10.7967L5.38697 9.86065Z"
        :stroke="getInput"
        stroke-width="1.5"
      />
      <path
        d="M23.8702 14.4217L24.6117 14.9994L14.9993 22.4757L5.38763 15L6.1161 14.4334L14.527 20.9726L14.9874 21.3306L15.4478 20.9725L23.8702 14.4217Z"
        :stroke="getMiddle"
        stroke-width="1.5"
      />
      <path
        d="M23.8704 19.5626L24.6114 20.1395L14.9993 27.6156L5.38763 20.1398L6.1161 19.5732L14.527 26.1125L14.9874 26.4704L15.4477 26.1124L23.8704 19.5626Z"
        :stroke="getOutput"
        stroke-width="1.5"
      />
      <rect x="5.67007" y="0.856468" width="0.601852" height="5.41667" rx="0.300926" fill="white" :stroke="color" stroke-width="0.601852" />
      <rect
        x="3.26382"
        y="3.8657"
        width="0.601852"
        height="5.41667"
        rx="0.300926"
        transform="rotate(-90 3.26382 3.8657)"
        fill="white"
        :stroke="color"
        stroke-width="0.601852"
      />
    </g>
    <defs>
      <clipPath id="clip0_3213:3118">
        <rect width="28.8889" height="28.8889" fill="white" transform="translate(0.554688 0.555542)" />
      </clipPath>
    </defs>
  </svg>
</template>

<script>
export default {
  name: 'd-icon-layer',
  props: {
    color: {
      type: String,
      default: '',
    },
    typeBlock: {
      type: String,
      default: '',
    },
  },
  computed: {
    getInput() {
      return this.typeBlock === 'input' ? this.color : '#6C7883';
    },
    getMiddle() {
      return this.typeBlock === 'middle' ? this.color : '#6C7883';
    },
    getOutput() {
      return this.typeBlock === 'output' ? this.color : '#6C7883';
    },
  },
};
</script>