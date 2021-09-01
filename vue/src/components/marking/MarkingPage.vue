<template>
  <div class="board">
    <div class="wrapper">
      <p>
        Lorem ipsum dolor sit amet consectetur adipisicing elit. Inventore cum, tempore ex excepturi quis magnam animi
        odit minima asperiores dicta natus quibusdam corporis sed ipsa nisi possimus eum itaque dolorem!
      </p>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';

export default {
  components: {},
  data: () => ({
    hight: 0,
  }),
  computed: {
    ...mapGetters({}),
  },
  methods: {},
};
</script>

<style lang="scss" scoped>
.board {
  flex-shrink: 1;
  width: 100%;
  background-color: #17212b;
}
</style>