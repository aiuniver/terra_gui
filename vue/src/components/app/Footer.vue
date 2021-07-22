<template>
  <div>
    <footer class="footer">
      <div class="message">
        <div class="wrapper">
          <span :class="color">
            {{ message }}
          </span>
        </div>
      </div>
      <div class="progress">
        <div class="wrapper">
          <i :style="{ width: progress + '%'}"><span></span></i><span></span>
        </div>
      </div>
      <div :class="`state ${protsessor}`">
        <div class="wrapper">{{ protsessor }}</div>
      </div>
      <div class="copyright">
        {{ `Copyright © «Университет искусственного интеллекта», ${new Date().getFullYear()}` }}
      </div>
    </footer>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
export default {
  computed: {
    ...mapGetters({
        message: 'messages/getMessage',
        color: 'messages/getColor',
        protsessor: 'messages/getProtsessor',
        progress: 'messages/getProgress',
    }),
  },
};
</script>

<style scoped>
.error {
  color: #FFB054 !important;
}
.success {
  color: #0f0 !important;
}
.footer {
  border-top: solid 1px #0e1621;
}
.state {
  text-transform: uppercase;
}
.copyright {
    position: fixed;
    width: 100%;
    left: 0;
    bottom: 0;
    z-index: 901;
    user-select: none;
    line-height: 30px;
    padding: 0 10px;
    color: #A7BED3;
    font-size: .6875rem;
    text-align: right;
    background-color: #0e1621;
}
</style>