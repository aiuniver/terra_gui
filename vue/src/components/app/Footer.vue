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
    </footer>
    <section id="copyright">
      <div class="container">
        {{ `Copyright © «Университет искусственного интеллекта», ${new Date().getFullYear()}` }}
      </div>
    </section>
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
</style>